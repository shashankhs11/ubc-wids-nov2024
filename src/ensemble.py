import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(train_path, test_path):
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Drop unnecessary columns and set index
    if 'Unnamed: 0' in train_df.columns:
        train_df.drop('Unnamed: 0', axis=1, inplace=True)
    if 'Unnamed: 0' in test_df.columns:
        test_df.drop('Unnamed: 0', axis=1, inplace=True)
    
    train_df.set_index('id', inplace=True)
    test_df.set_index('id', inplace=True)

    for df in [train_df, test_df]:
        for col in ['host_response_rate', 'host_acceptance_rate']:
            if col in df.columns:
                df[col] = df[col].str.replace('%', '').astype(float)  # Convert to numeric
    
    return train_df, test_df

def extract_features_from_name(df):
    # Extract features from name column
    def extract_rating(parts):
        for part in parts:
            if '★' in part:
                try:
                    return float(part.replace('★', '').strip())
                except ValueError:
                    continue
        return 0

    def is_property_new(parts):
        for part in parts:
            if 'new' in part.lower():
                return 1
        return 0

    def extract_bedrooms(parts):
        for part in parts:
            if 'Studio' in part:
                return 0
            elif 'bedroom' in part:
                try:
                    return int(part.split()[0])
                except ValueError:
                    continue
        return 0

    def extract_beds(parts):
        for part in parts:
            if 'bed' in part:
                try:
                    return int(part.split()[0])
                except ValueError:
                    continue
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

    df["split_parts"] = df["name"].str.split("·")
    df["bedrooms"] = df["split_parts"].apply(extract_bedrooms)
    df["beds"] = df["split_parts"].apply(extract_beds)
    df["baths"] = df["split_parts"].apply(extract_baths)
    df["is_bath_private"] = df["split_parts"].apply(is_private_bath).astype(int)
    df["is_bath_shared"] = df["split_parts"].apply(is_shared_bath).astype(int)
    df["overall_rating"] = df["split_parts"].apply(extract_rating)
    df["is_new_property"] = df["split_parts"].apply(is_property_new).astype(int)
    
    df.drop('split_parts', axis=1, inplace=True)
    return df

def process_neighborhood_overview(df, tfidf_vectorizer=None):
    def clean_text(text):
        if pd.isna(text):
            return ''
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['cleaned_neighborhood_overview'] = df['neighborhood_overview'].apply(clean_text)
    df['cleaned_neighborhood_overview'].fillna('no description available', inplace=True)
    
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_neighborhood_overview'])
    else:
        tfidf_matrix = tfidf_vectorizer.transform(df['cleaned_neighborhood_overview'])
    
    # Convert to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                           columns=[f'tfidf_{i}' for i in range(100)],
                           index=df.index)
    
    return pd.concat([df, tfidf_df], axis=1), tfidf_vectorizer

def create_feature_engineering(df):
    # Clean price column
    df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Create revenue features
    df['minimum_monthly_revenue'] = df['minimum_nights'] * df['price']
    df['maximum_monthly_revenue'] = df.apply(
        lambda row: (row['maximum_nights'] / 30) * row['price'] 
        if row['maximum_nights'] > 30 
        else row['maximum_nights'] * row['price'],
        axis=1
    )
    df['revenue_per_booking'] = df['accommodates'] * df['price']

    # Create availability ratios
    availability_days = [30, 60, 90, 365]
    for days in availability_days:
        # Compute availability ratio
        df[f'availability_ratio_{days}'] = df[f'availability_{days}'] / days
        # Compute monthly revenue
        factor = 30 if days != 30 else 1
        df[f'monthly_revenue_{days}'] = df[f'availability_ratio_{days}'] * df['price'] * factor
    
    return df

def prepare_features(df):
    # Columns to drop
    cols_to_drop = ['host_id', 'host_name', 'neighbourhood', 'latitude', 'longitude', 
                    'amenities', 'name', 'neighborhood_overview', 'cleaned_neighborhood_overview']
    
    # Categorical columns for one-hot encoding
    categorical_features = ['host_response_time', 'neighbourhood_cleansed', 
                          'property_type', 'room_type', 'host_is_superhost']
    
    # Numerical columns for scaling
    numerical_features = ['host_response_rate', 'host_acceptance_rate', 
                         'host_listings_count', 'host_total_listings_count',
                         'accommodates', 'beds', 'minimum_nights', 'maximum_nights',
                         'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm',
                         'number_of_reviews', 'number_of_reviews_ltm',
                         'review_scores_rating', 'review_scores_accuracy',
                         'review_scores_cleanliness', 'review_scores_checkin',
                         'review_scores_communication', 'review_scores_location',
                         'review_scores_value', 'calculated_host_listings_count',
                         'reviews_per_month']
    
    robust_features = ['minimum_monthly_revenue','maximum_monthly_revenue', 
                       'revenue_per_booking', 'price']
    
    # Add availability ratios and monthly revenue columns
    for days in [30, 60, 90, 365]:
        numerical_features.extend([f'availability_ratio_{days}', f'monthly_revenue_{days}'])
    
    # Add engineered features from name
    numerical_features.extend(['bedrooms', 'baths', 'overall_rating'])
    categorical_features.extend(['is_bath_private', 'is_bath_shared', 'is_new_property'])
    
    # Add TF-IDF features
    tfidf_features = [col for col in df.columns if col.startswith('tfidf_')]
    numerical_features.extend(tfidf_features)
    
    # Drop specified columns
    df = df.drop(cols_to_drop, axis=1)
    
    return df, numerical_features, categorical_features, robust_features

def create_model_pipeline(numerical_features, categorical_features, robust_features):
    # Create preprocessors
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    robust_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
    ])

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('robust', robust_transformer, robust_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define parameter distributions for each model
    param_distributions = {
    'RandomForest': {
        'regressor__n_estimators': randint(100, 500),
        'regressor__max_depth': [None] + list(range(10, 50, 5)),
        'regressor__min_samples_split': randint(2, 20),
        'regressor__min_samples_leaf': randint(1, 10),
        'regressor__max_features': ['sqrt', 'log2', None]
    },
    'GradientBoosting': {
        'regressor__n_estimators': randint(100, 500),
        'regressor__learning_rate': uniform(0.01, 0.3),
        'regressor__max_depth': randint(3, 10),
        'regressor__min_samples_split': randint(2, 20),
        'regressor__min_samples_leaf': randint(1, 10),
        'regressor__subsample': uniform(0.6, 0.4)
    },
    'XGBoost': {
        'regressor__n_estimators': randint(100, 500),
        'regressor__learning_rate': uniform(0.01, 0.3),
        'regressor__max_depth': randint(3, 10),
        'regressor__min_child_weight': randint(1, 7),
        'regressor__subsample': uniform(0.6, 0.4),
        'regressor__colsample_bytree': uniform(0.6, 0.4)
    },
    'ElasticNet': {
        'regressor__alpha': uniform(0.0001, 1.0),
        'regressor__l1_ratio': uniform(0, 1),
        'regressor__max_iter': [2000]
    },
    'SVM': {
        'regressor__C': uniform(0.1, 10),
        'regressor__epsilon': uniform(0.01, 1),
        'regressor__kernel': ['linear', 'rbf'],
        'regressor__gamma': ['scale', 'auto']
    }
}

    # Create base model pipelines
    base_models = {
    'RandomForest': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ]),
    'GradientBoosting': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(random_state=42))
    ]),
    'ElasticNet': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet(random_state=42))
    ]),
    'SVM': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', SVR())
    ])
}

    # Create RandomizedSearchCV for each model
    models = {}
    for name, pipeline in base_models.items():
        models[name] = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions[name],
            n_iter=30,  # Number of parameter settings sampled
            cv=5,       # Number of cross-validation folds
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,  # Use all available cores
            random_state=42,
            verbose=1
        )

    return models

def evaluate_models_and_create_submissions(models, X_train, y_train, X_test, test_df):
    results = {}
    best_params = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} with RandomizedSearchCV...")
        
        # Fit model with randomized search on the full training data
        model.fit(X_train, y_train)
        
        # Store best parameters
        best_params[name] = model.best_params_
        print(f"\nBest parameters for {name}:")
        for param, value in model.best_params_.items():
            print(f"{param}: {value}")
        
        # Evaluate model on the training data
        train_pred = model.predict(X_train)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mse = mean_squared_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        # Store results
        results[name] = {
            'Train RMSE': train_rmse,
            'Train MSE': train_mse,
            'Train R2': train_r2,
            'Best CV Score': -model.best_score_  # Convert back from negative RMSE
        }
        
        # Create submission file
        print(f"Creating submission for {name}...")
        
        # Retrain best model on full training data (optional if model.best_estimator_ is already trained)
        model.best_estimator_.fit(X_train, y_train)
        
        # Predict on test data
        test_predictions = model.best_estimator_.predict(X_test)
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'id': test_df.index,
            'monthly_revenue': test_predictions
        })
        
        # Save submission
        submission.to_csv(f'submission_{name.lower()}_tuned.csv', index=False)
        print(f"Saved submission_{name.lower()}_tuned.csv")
    
    return results, best_params

def main(selected_model=None):
    # Load data
    train_df, test_df = load_and_preprocess_data('input/train.csv', 'input/test.csv')
    
    # Process name column
    print("Processing name column...")
    train_df = extract_features_from_name(train_df)
    test_df = extract_features_from_name(test_df)
    
    # Process neighborhood overview
    print("Processing neighborhood overview...")
    train_df, tfidf_vectorizer = process_neighborhood_overview(train_df)
    test_df, _ = process_neighborhood_overview(test_df, tfidf_vectorizer)
    
    # Create engineered features
    print("Creating engineered features...")
    train_df = create_feature_engineering(train_df)
    test_df = create_feature_engineering(test_df)
    
    # Prepare features
    print("Preparing features...")
    train_df, numerical_features, categorical_features = prepare_features(train_df)
    test_df, _, _ = prepare_features(test_df)
    
    # train_df.to_csv('preprocessed.csv')
    # Split features and target
    X_train = train_df.drop('monthly_revenue', axis=1)
    y_train = train_df['monthly_revenue']
    X_test = test_df
    
    print("Creating model pipelines with RandomizedSearchCV...")
    models = create_model_pipeline(numerical_features, categorical_features, robust_features)
    
    # Filter for the selected model
    if selected_model:
        if isinstance(selected_model, list):  # If multiple models are provided
            models = {name: model for name, model in models.items() if name in selected_model}
        else:
            models = {selected_model: models[selected_model]}

    
    # Evaluate models and create submissions
    results, best_params = evaluate_models_and_create_submissions(
        models, X_train, y_train, X_test, test_df
    )
    
    # Print results
    print("\nFinal Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        print("\nBest Parameters:")
        for param, value in best_params[model_name].items():
            print(f"{param}: {value}")

class BaggedXGBoostRegressor:
    def __init__(
        self,
        n_cv_seeds=3,
        n_model_seeds=3,
        n_folds=5,
        base_params=None,
        random_state=42
    ):
        self.n_cv_seeds = n_cv_seeds
        self.n_model_seeds = n_model_seeds
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = []
        self.cv_scores = []
        
        # Default XGBoost parameters
        self.base_params = base_params if base_params else {
            'n_estimators': 500,
            'learning_rate': 0.01,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': random_state
        }

    def _create_preprocessor(self, numerical_features, categorical_features, robust_features):
        return ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numerical_features),
                ('robust', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', RobustScaler())
                ]), robust_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), categorical_features)
            ])

    def _create_pipeline(self, preprocessor, xgb_params):
        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(**xgb_params))
        ])

    def fit(self, X, y, numerical_features, categorical_features, robust_features):
        """
        Fit multiple XGBoost models using different CV splits and random seeds
        """
        self.models = []
        self.cv_scores = []
        
        # Create preprocessor
        preprocessor = self._create_preprocessor(
            numerical_features, categorical_features, robust_features
        )

        # Iterate over CV random seeds
        for cv_seed in range(self.n_cv_seeds):
            print(f"\nCV Split Seed {cv_seed + 1}/{self.n_cv_seeds}")
            
            # Create CV splits
            kf = KFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state + cv_seed
            )
            
            # Iterate over model random seeds
            for model_seed in range(self.n_model_seeds):
                print(f"Model Seed {model_seed + 1}/{self.n_model_seeds}")
                
                fold_models = []
                fold_scores = []
                
                # Iterate over folds
                for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                    print(f"Training Fold {fold}/{self.n_folds}")
                    
                    # Split data
                    X_train_fold = X.iloc[train_idx]
                    y_train_fold = y.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    y_val_fold = y.iloc[val_idx]
                    
                    # Create and train model
                    xgb_params = self.base_params.copy()
                    xgb_params['random_state'] = self.random_state + model_seed
                    
                    pipeline = self._create_pipeline(preprocessor, xgb_params)
                    pipeline.fit(X_train_fold, y_train_fold)
                    
                    # Evaluate
                    val_pred = pipeline.predict(X_val_fold)
                    rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
                    
                    fold_models.append(pipeline)
                    fold_scores.append(rmse)
                    
                    print(f"Fold {fold} RMSE: {rmse:.4f}")
                
                # Store models and scores
                self.models.extend(fold_models)
                self.cv_scores.extend(fold_scores)
                
                print(f"Average RMSE for Model Seed {model_seed + 1}: "
                      f"{np.mean(fold_scores):.4f}")
        
        return self

    def predict(self, X):
        """
        Make predictions using all trained models
        """
        predictions = []
        weights = 1 / np.array(self.cv_scores)
        weights = weights / np.sum(weights)
        
        # Get predictions from all models
        for model, weight in zip(self.models, weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        # Return weighted average
        return np.sum(predictions, axis=0)

def train_bagged_xgboost(X_train, y_train, X_test, test_df, 
                        numerical_features, categorical_features, robust_features,
                        n_cv_seeds=3, n_model_seeds=3, n_folds=5):
    """
    Train bagged XGBoost model and create submission
    """
    print("Training Bagged XGBoost...")
    
    # Initialize and train bagged model
    bagged_model = BaggedXGBoostRegressor(
        n_cv_seeds=n_cv_seeds,
        n_model_seeds=n_model_seeds,
        n_folds=n_folds
    )
    
    bagged_model.fit(
        X_train, y_train,
        numerical_features, categorical_features, robust_features
    )
    
    # Make predictions
    predictions = bagged_model.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df.index,
        'monthly_revenue': predictions
    })
    
    # Save submission
    submission_file = 'submission_bagged_xgboost.csv'
    submission.to_csv(submission_file, index=False)
    print(f"\nSaved submission to {submission_file}")
    
    # Print summary statistics
    print("\nModel Performance Summary:")
    print(f"Number of base models: {len(bagged_model.models)}")
    print(f"Average RMSE across all folds: {np.mean(bagged_model.cv_scores):.4f}")
    print(f"Best fold RMSE: {np.min(bagged_model.cv_scores):.4f}")
    print(f"Worst fold RMSE: {np.max(bagged_model.cv_scores):.4f}")
    
    return bagged_model, submission

# Modified main function to use bagged XGBoost
def main_bagged(n_cv_seeds=3, n_model_seeds=3, n_folds=5):
    # Load and preprocess data
    train_df, test_df = load_and_preprocess_data('input/train.csv', 'input/test.csv')
    
    # Process name column
    print("Processing name column...")
    train_df = extract_features_from_name(train_df)
    test_df = extract_features_from_name(test_df)
    
    # Process neighborhood overview
    print("Processing neighborhood overview...")
    train_df, tfidf_vectorizer = process_neighborhood_overview(train_df)
    test_df, _ = process_neighborhood_overview(test_df, tfidf_vectorizer)
    
    # Create engineered features
    print("Creating engineered features...")
    train_df = create_feature_engineering(train_df)
    test_df = create_feature_engineering(test_df)
    
    # Prepare features
    print("Preparing features...")
    train_df, numerical_features, categorical_features, robust_features = prepare_features(train_df)
    test_df, _, _, _ = prepare_features(test_df)
    
    # Split features and target
    X_train = train_df.drop('monthly_revenue', axis=1)
    y_train = train_df['monthly_revenue']
    X_test = test_df
    
    # Train bagged model and create submission
    model, submission = train_bagged_xgboost(
        X_train, y_train, X_test, test_df,
        numerical_features, categorical_features, robust_features,
        n_cv_seeds, n_model_seeds, n_folds
    )
    
    return model, submission

if __name__ == "__main__":
    # main("XGBoost")
    main_bagged(n_cv_seeds=3, n_model_seeds=3, n_folds=5)
    # main_ensemble(n_models=5)