import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform, randint
from transformers import AutoTokenizer, AutoModel
import torch
import xgboost as xgb
from sklearn.linear_model import LassoCV, RidgeCV


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Keep the existing feature extraction functions
def extract_bedrooms(parts):
    for part in parts:
        if 'Studio' in part:
            return 0
        elif 'bedroom' in part:
            return int(part.split()[0])
    return 0

def extract_beds(parts):
    for part in parts:
        if 'bed' in part:
            return int(part.split()[0])
    return 0

def extract_baths(parts):
    for part in parts:
        if 'half-bath' in part.lower():
            return 0.5
        if 'bath' in part.lower():
            try:
                return float(part.split()[0])
            except ValueError:
                continue
    return 0

def is_private_bath(parts):
    for part in parts:
        if 'private' in part.lower() and 'bath' in part.lower():
            return 1
    return 0

def is_shared_bath(parts):
    for part in parts:
        if 'shared' in part.lower() and 'bath' in part.lower():
            return 1
    return 0

def preprocess_data(train_df, test_df=None):
    # Same preprocessing as before
    drop_features = ['host_id', 'host_name', 'latitude', 'longitude', 'amenities', 
                    'bathrooms', 'bedrooms', 'host_listings_count', 
                    'host_total_listings_count', 'neighbourhood', 'name']
    
    train_df = train_df.copy()
    test_df = test_df.copy() if test_df is not None else None
    
    train_name = train_df['name']
    test_name = test_df['name'] if test_df is not None else None
    
    for df in [train_df] + ([test_df] if test_df is not None else []):
        features_to_drop = [f for f in drop_features if f in df.columns]
        df.drop(columns=features_to_drop, inplace=True)
    
    for df, name_series in [(train_df, train_name)] + ([(test_df, test_name)] if test_df is not None else []):
        df["split_parts"] = name_series.str.split("·")
        df["bedrooms"] = df["split_parts"].apply(extract_bedrooms)
        df["beds"] = df["split_parts"].apply(extract_beds)
        df["baths"] = df["split_parts"].apply(extract_baths)
        df["is_bath_private"] = df["split_parts"].apply(is_private_bath).astype(int)
        df["is_bath_shared"] = df["split_parts"].apply(is_shared_bath).astype(int)
        df.drop('split_parts', axis=1, inplace=True)
        
        if 'host_is_superhost' in df.columns:
            df['host_is_superhost'] = (df['host_is_superhost'] == 't').astype(int)
        if 'instant_bookable' in df.columns:
            df['instant_bookable'] = (df['instant_bookable'] == 't').astype(int)
    
    train_df['price'] = train_df['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    if test_df is not None:
        test_df['price'] = test_df['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)

    return train_df, test_df

class BERTEmbedding:
    def __init__(self, model_name='bert-base-uncased'):
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def get_embeddings(self, texts, max_length=512):
        embeddings = []
        with torch.no_grad():
            for text in texts:
                if pd.isna(text):
                    embeddings.append(torch.zeros(768, device=self.device))
                    continue
                    
                inputs = self.tokenizer(text, 
                                      return_tensors="pt",
                                      max_length=max_length,
                                      padding=True,
                                      truncation=True)
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                embeddings.append(embedding)
                
        stacked = torch.stack(embeddings)
        # Move back to CPU for numpy conversion
        return stacked.cpu()
    

class MLPredictor:
    def __init__(self, model_type='xgboost', n_iter=50):
        """
        Initialize MLPredictor
        
        Args:
            model_type (str): Type of model to use
            n_iter (int): Number of parameter settings sampled in random search
        """
        self.model_type = model_type
        self.n_iter = n_iter
        self.model = None
        self.bert_embedder = BERTEmbedding()
        self.preprocessor = None
        self.best_params = None
        self.train_embeddings = None
        self.test_embeddings = None
        
    def _get_model_and_param_dist(self):
        """Define model and parameter distributions for random search"""
        if self.model_type == 'xgboost':
            model = xgb.XGBRegressor(
                random_state=42,
                tree_method='gpu_hist' if torch.backends.mps.is_available() or torch.cuda.is_available() else 'hist'
            )
            param_dist = {
                'n_estimators': randint(100, 1500),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'min_child_weight': randint(1, 7),
                'gamma': uniform(0, 0.5)
            }
        elif self.model_type == 'gradientboosting':
            model = GradientBoostingRegressor(random_state=42)
            param_dist = {
                'n_estimators': randint(100, 1500),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'subsample': uniform(0.6, 0.4),
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 5),
                'max_features': ['sqrt', 'log2', None]
            }
        elif self.model_type == 'randomforest':
            model = RandomForestRegressor(random_state=42)
            param_dist = {
                'n_estimators': randint(100, 1500),
                'max_depth': randint(10, 30),
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 5),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        return model, param_dist
    
    def create_features(self, train_df: pd.DataFrame, test_df=None, fit=True):
        cat_features = [
            'host_response_time', 'host_is_superhost', 'neighbourhood_cleansed',
            'property_type', 'room_type', 'accommodates', 'beds',
            'instant_bookable', 'calculated_host_listings_count',
            'is_bath_private', 'is_bath_shared'
        ]

        num_features = [
            'price', 'minimum_nights', 'maximum_nights', 'minimum_nights_avg_ntm',
            'maximum_nights_avg_ntm', 'availability_30', 'availability_60',
            'availability_90', 'availability_365', 'number_of_reviews',
            'number_of_reviews_ltm', 'review_scores_rating', 'review_scores_accuracy',
            'review_scores_cleanliness', 'review_scores_checkin',
            'review_scores_communication', 'review_scores_location',
            'review_scores_value', 'reviews_per_month', 'bedrooms', 'beds', 'baths'
        ]

        # Get available features
        available_cat_features = [f for f in cat_features if f in train_df.columns]
        available_num_features = [f for f in num_features if f in train_df.columns]

        if not available_cat_features and not available_num_features:
            raise ValueError("No categorical or numerical features are available for preprocessing.")

        # Initialize preprocessor if it doesn't exist
        if self.preprocessor is None:
            print("Initializing preprocessor...")
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', Pipeline([
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
                    ]), available_cat_features),
                    ('num', Pipeline([
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler())
                    ]), available_num_features)
                ]
            )

        # Fit or transform data
        if fit:
            train_features = self.preprocessor.fit_transform(train_df)
        else:
            train_features = self.preprocessor.transform(train_df)

        if test_df is not None:
            test_features = self.preprocessor.transform(test_df)
            return train_features, test_features

        return train_features, None

    
    def fit(self, train_df: pd.DataFrame, y: np.ndarray) -> 'MLPredictor':
        # Split data into train and validation sets
        train_data, val_data, train_y, val_y = train_test_split(
            train_df, y, test_size=0.2, random_state=42
        )
        
        # Create features for training
        X_train, _ = self.create_features(train_data, fit=True)
        X_val, _ = self.create_features(val_data, fit=False)
        
        # Get model and parameter distributions
        base_model, param_dist = self._get_model_and_param_dist()
        
        # Perform random search with cross-validation
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=self.n_iter,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1 if self.model_type != 'xgboost' else 1,  # Use single job for XGBoost with GPU
            verbose=1,
            random_state=42
        )
        
        # Fit the model
        random_search.fit(X_train, train_y)
        
        # Save best model and parameters
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        
        # Print results
        print(f"\nBest parameters for {self.model_type}:")
        for param, value in self.best_params.items():
            print(f"{param}: {value}")
            
        print(f"\nBest CV RMSE: {-random_search.best_score_:.2f}")
        
        # Evaluate on validation set
        val_pred = self.model.predict(X_val)
        val_rmse = np.sqrt(np.mean((val_y - val_pred) ** 2))
        print(f"Validation RMSE: {val_rmse:.2f}")
        
        return self
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X_test = self.create_features(test_df, fit=False)[0]
        return self.model.predict(X_test)

if __name__ == "__main__":
    # Enable Metal backend for PyTorch
    if torch.backends.mps.is_available():
        print("MPS (Metal) backend is available")
    elif torch.cuda.is_available():
        print("CUDA backend is available")
    else:
        print("Using CPU backend")
    
    # Load data
    train_df = pd.read_csv('input/train.csv')
    test_df = pd.read_csv('input/test.csv')
    
    train_df.set_index('id', inplace=True)
    test_df.set_index('id', inplace=True)
    
    # Preprocess data
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Train and predict with different models
    models = ['xgboost', 'gradientboosting', 'randomforest']
    
    for model_type in models:
        print(f"\nTraining {model_type}...")
        predictor = MLPredictor(model_type=model_type, n_iter=50)
        predictor.fit(train_df, train_df['monthly_revenue'].values)
        
        predictions = predictor.predict(test_df)
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_df.index,
            'monthly_revenue': predictions
        })
        submission.to_csv(f'submission_{model_type}.csv', index=False)