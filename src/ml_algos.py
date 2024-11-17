import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from transformers import AutoTokenizer, AutoModel
import torch
import xgboost as xgb
from sklearn.linear_model import LassoCV, RidgeCV

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

class BERTEmbedding:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def get_embeddings(self, texts, max_length=512):
        embeddings = []
        with torch.no_grad():
            for text in texts:
                if pd.isna(text):
                    embeddings.append(torch.zeros(768))
                    continue
                    
                inputs = self.tokenizer(text, 
                                      return_tensors="pt",
                                      max_length=max_length,
                                      padding=True,
                                      truncation=True)
                
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                embeddings.append(embedding)
                
        return torch.stack(embeddings)

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

class MLPredictor:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.bert_embedder = BERTEmbedding()
        self.preprocessor = None
        
    def _get_model(self):
        if self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif self.model_type == 'gradientboosting':
            return GradientBoostingRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'randomforest':
            return RandomForestRegressor(
                n_estimators=1000,
                max_depth=15,
                random_state=42
            )
        elif self.model_type == 'lasso':
            return LassoCV(cv=5)
        elif self.model_type == 'ridge':
            return RidgeCV(cv=5)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def create_features(self, train_df, test_df=None):
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
        
        # Get BERT embeddings
        neighborhood_overview_train = train_df.get('neighborhood_overview', pd.Series([''] * len(train_df)))
        train_embeddings = self.bert_embedder.get_embeddings(neighborhood_overview_train)
        
        if test_df is not None:
            neighborhood_overview_test = test_df.get('neighborhood_overview', pd.Series([''] * len(test_df)))
            test_embeddings = self.bert_embedder.get_embeddings(neighborhood_overview_test)
        
        # Drop text column after getting embeddings
        if 'neighborhood_overview' in train_df.columns:
            train_df.drop('neighborhood_overview', axis=1, inplace=True)
        if test_df is not None and 'neighborhood_overview' in test_df.columns:
            test_df.drop('neighborhood_overview', axis=1, inplace=True)
        
        # Filter available features
        available_cat_features = [f for f in cat_features if f in train_df.columns]
        available_num_features = [f for f in num_features if f in train_df.columns]
        
        # Create preprocessor if not already created
        if self.preprocessor is None:
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
                ])
        
        # Transform features
        train_features = self.preprocessor.fit_transform(train_df)
        if test_df is not None:
            test_features = self.preprocessor.transform(test_df)
        
        # Combine with BERT embeddings
        train_combined = np.hstack([train_features, train_embeddings.numpy()])
        if test_df is not None:
            test_combined = np.hstack([test_features, test_embeddings.numpy()])
            return train_combined, test_combined
        
        return train_combined
    
    def fit(self, train_df, y):
        X = self.create_features(train_df)
        self.model = self._get_model()
        self.model.fit(X, y)
        
        # Print cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_root_mean_squared_error')
        print(f"Cross-validation RMSE: {-cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        return self
    
    def predict(self, test_df):
        X_train, X_test = self.create_features(train_df, test_df)
        return self.model.predict(X_test)

if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv('input/train.csv')
    test_df = pd.read_csv('input/test.csv')
    
    train_df.set_index('id', inplace=True)
    test_df.set_index('id', inplace=True)
    
    # Preprocess data
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Train and predict with different models
    models = ['xgboost', 'gradientboosting', 'randomforest', 'lasso', 'ridge']
    
    for model_type in models:
        print(f"\nTraining {model_type}...")
        predictor = MLPredictor(model_type=model_type)
        predictor.fit(train_df, train_df['monthly_revenue'].values)
        
        predictions = predictor.predict(test_df)
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_df.index,
            'monthly_revenue': predictions
        })
        submission.to_csv(f'submission_{model_type}.csv', index=False)