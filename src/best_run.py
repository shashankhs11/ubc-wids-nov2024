import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def load_data(train_path, test_path=None):
    """Load and perform initial data cleaning"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path else None
    
    # Drop unnecessary columns
    columns_to_drop = ['bathrooms', 'bedrooms', 'Unnamed: 0'] if 'Unnamed: 0' in train_df.columns else ['bathrooms', 'bedrooms']
    for col in columns_to_drop:
        if col in train_df.columns:
            train_df.drop(col, axis=1, inplace=True)
        if test_df is not None and col in test_df.columns:
            test_df.drop(col, axis=1, inplace=True)
    
    return train_df, test_df

def extract_features_from_name(name):
    """Extract features from listing name"""
    property_type_pattern = r"^(.*?) in Vancouver"
    rating_pattern = r"★([0-9.]+|New)"
    bedrooms_pattern = r"(\d+) bedroom"
    beds_pattern = r"(\d+) bed"
    baths_pattern = r"(\d+) bath"
    
    property_type = re.search(property_type_pattern, str(name))
    rating = re.search(rating_pattern, str(name))
    bedrooms = re.search(bedrooms_pattern, str(name))
    beds = re.search(beds_pattern, str(name))
    baths = re.search(baths_pattern, str(name))
    
    return {
        'extracted_property_type': property_type.group(1) if property_type else "Other",
        'extracted_rating': float(rating.group(1)) if rating and rating.group(1).replace('.', '', 1).isdigit() else None,
        'extracted_bedrooms': int(bedrooms.group(1)) if bedrooms else (0 if 'Studio' in str(name) else None),
        'extracted_beds': int(beds.group(1)) if beds else None,
        'extracted_baths': int(baths.group(1)) if baths else None
    }

def process_neighborhood_features(text):
    """Process neighborhood overview text"""
    # Clean HTML tags
    cleaned_text = re.sub(r'<.*?>', '', str(text))
    
    # Keywords presence
    keywords = ['park', 'restaurant', 'shopping', 'beach', 'quiet', 'transit', 'lively', 'trendy']
    keyword_features = {f'has_{keyword}': int(bool(re.search(r'\b' + keyword + r'\b', cleaned_text.lower())))
                       for keyword in keywords}
    
    # Sentiment analysis
    try:
        blob = TextBlob(cleaned_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    except:
        polarity = 0
        subjectivity = 0
    
    sentiment_features = {
        'polarity': polarity,
        'subjectivity': subjectivity,
        'review_length': len(cleaned_text.split())
    }
    
    return {**keyword_features, **sentiment_features}

def safe_encode_categorical(df, column, default_value='unknown'):
    """Safely encode categorical variables"""
    # Ensure the column contains string values
    df[column] = df[column].fillna(default_value).astype(str)
    
    # If the column contains lists (detected by presence of brackets), take first value
    df[column] = df[column].apply(lambda x: x.split('[')[0].split(',')[0].strip() if '[' in x else x)
    
    # Create label encoder
    le = LabelEncoder()
    encoded_values = le.fit_transform(df[column])
    
    return encoded_values, le

def feature_engineering(df, is_training=True):
    """Comprehensive feature engineering pipeline"""
    df_processed = df.copy()
    
    # Extract features from name
    name_features = pd.DataFrame([extract_features_from_name(name) for name in df_processed['name']])
    df_processed = pd.concat([df_processed, name_features], axis=1)
    
    # Process neighborhood features
    df_processed['neighborhood_overview'] = df_processed['neighborhood_overview'].fillna("No description provided")
    neighborhood_features = pd.DataFrame([
        process_neighborhood_features(text) for text in df_processed['neighborhood_overview']
    ])
    df_processed = pd.concat([df_processed, neighborhood_features], axis=1)
    
    # Handle missing values and convert datatypes
    df_processed['is_new_listing'] = df_processed['extracted_rating'].isnull().astype(int)
    df_processed['rating'] = df_processed['extracted_rating'].fillna(0)
    
    # Process response time and rates
    if 'host_response_time' in df_processed.columns:
        df_processed['host_response_time'] = df_processed['host_response_time'].fillna('not_specified')
    
    # Process percentage columns
    for col in ['host_response_rate', 'host_acceptance_rate']:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].apply(lambda x: 
                float(str(x).strip('%')) / 100 if pd.notna(x) and isinstance(x, str) else float(x) if pd.notna(x) else 0.0
            )
    
    # Convert boolean columns
    boolean_columns = ['host_is_superhost']
    for col in boolean_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map({'t': 1, 'f': 0, True: 1, False: 0}).fillna(0)
    
    # Process amenities
    if 'amenities' in df_processed.columns:
        df_processed['amenities_count'] = df_processed['amenities'].str.len()
    
    # Price processing
    if 'price' in df_processed.columns:
        df_processed['price'] = df_processed['price'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Review scores processing
    review_cols = [col for col in df_processed.columns if col.startswith('review_scores_')]
    if review_cols:
        df_processed[review_cols] = df_processed[review_cols].fillna(df_processed[review_cols].mean())
        df_processed['avg_review_score'] = df_processed[review_cols].mean(axis=1)
    
    # Categorical encoding
    categorical_cols = ['room_type', 'extracted_property_type', 'neighbourhood_cleansed']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df_processed.columns:
            try:
                encoded_values, le = safe_encode_categorical(df_processed, col)
                df_processed[f'{col}_encoded'] = encoded_values
                if is_training:
                    label_encoders[col] = le
            except Exception as e:
                print(f"Warning: Could not encode column {col}: {str(e)}")
                continue
    
    return (df_processed, label_encoders) if is_training else df_processed

def prepare_features(df):
    """Prepare final feature set for modeling"""
    feature_groups = {
        'host_features': ['host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
                         'host_listings_count'],
        'property_features': ['accommodates', 'extracted_beds', 'price', 'amenities_count',
                            'extracted_property_type_encoded', 'room_type_encoded'],
        'location_features': ['neighbourhood_cleansed_encoded'] + 
                           [col for col in df.columns if col.startswith('has_')],
        'review_features': ['number_of_reviews', 'reviews_per_month', 'avg_review_score',
                           'rating', 'is_new_listing', 'polarity', 'subjectivity'],
        'availability_features': ['minimum_nights', 'maximum_nights', 'availability_365']
    }
    
    all_features = [feature for group in feature_groups.values() for feature in group]
    available_features = [col for col in all_features if col in df.columns]
    
    # Print available and missing features for debugging
    missing_features = [col for col in all_features if col not in df.columns]
    print("\nAvailable features:", len(available_features))
    print("Missing features:", missing_features)
    
    return df[available_features].fillna(0)

def train_model(X, y):
    """Train and evaluate models"""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = float('-inf')
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        r2 = r2_score(y_val, val_pred)
        
        print(f"\n{name} Results:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = model
    
    return best_model

def tune_model(X, y):
    """Hyperparameter tuning for Random Forest"""
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_model = RandomForestRegressor(random_state=42)
    rf_random = RandomizedSearchCV(
        rf_model, param_grid, n_iter=20, cv=5, 
        random_state=42, scoring='neg_mean_squared_error'
    )
    rf_random.fit(X, y)
    return rf_random.best_estimator_

class ValidationPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            'Random Forest': RandomForestRegressor(random_state=random_state),
            'Gradient Boosting': GradientBoostingRegressor(random_state=random_state),
            'Extra Trees': ExtraTreesRegressor(random_state=random_state),
            'XGBoost': XGBRegressor(random_state=random_state),
            'LightGBM': LGBMRegressor(random_state=random_state),
            'Lasso': LassoCV(random_state=random_state),
            'Ridge': RidgeCV(),
            'SVR': SVR(kernel='rbf')
        }
        self.best_model = None
        self.feature_importance = None
        
    def create_folds(self, X, y, n_splits=5):
        """Create stratified folds based on target variable quantiles"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        return [(train_idx, val_idx) for train_idx, val_idx in kf.split(X)]
    
    def evaluate_model(self, model, X, y, folds):
        """Evaluate model using k-fold cross validation"""
        scores = {
            'rmse': [],
            'mae': [],
            'r2': []
        }
        
        for fold_n, (train_idx, val_idx) in enumerate(folds):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            
            scores['rmse'].append(np.sqrt(mean_squared_error(y_val, pred)))
            scores['mae'].append(mean_absolute_error(y_val, pred))
            scores['r2'].append(r2_score(y_val, pred))
        
        return {metric: np.mean(values) for metric, values in scores.items()}
    
    def train_and_evaluate(self, X, y):
        """Train and evaluate all models using k-fold validation"""
        print("\nStarting model evaluation...")
        folds = self.create_folds(X, y)
        results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            try:
                scores = self.evaluate_model(model, X, y, folds)
                results[name] = scores
                print(f"Average RMSE: {scores['rmse']:.2f}")
                print(f"Average MAE: {scores['mae']:.2f}")
                print(f"Average R2: {scores['r2']:.4f}")
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
        
        # Select best model based on RMSE
        best_model_name = min(results.keys(), key=lambda k: results[k]['rmse'])
        self.best_model = self.models[best_model_name]
        print(f"\nBest model: {best_model_name}")
        
        return results
    
    def analyze_feature_importance(self, X, y):
        """Analyze and plot feature importance"""
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=self.feature_importance.head(15), 
                       x='importance', y='feature')
            plt.title('Top 15 Most Important Features')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()
            
            return self.feature_importance
        else:
            print("Selected model doesn't support feature importance analysis")
            return None
        

def main():
    # Load data
    train_df, test_df = load_data('input/train.csv', 'input/test.csv')
    
    print("Data loaded successfully")
    print(f"Train shape: {train_df.shape}")
    if test_df is not None:
        print(f"Test shape: {test_df.shape}")
    
    # Feature engineering
    print("\nStarting feature engineering...")
    train_processed, label_encoders = feature_engineering(train_df, is_training=True)
    if test_df is not None:
        test_processed = feature_engineering(test_df, is_training=False)
    print("Feature engineering completed")
    
    # Prepare features
    print("\nPreparing features...")
    X = prepare_features(train_processed)
    y = train_processed['monthly_revenue']
    
    print("\nFeature matrix shape:", X.shape)
    print("Features used in the model:", X.columns.tolist())
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Initialize and run validation pipeline
    pipeline = ValidationPipeline()
    results = pipeline.train_and_evaluate(X_train, y_train)
    
    # Analyze feature importance
    feature_importance = pipeline.analyze_feature_importance(X_train, y_train)
    
    # Final evaluation on validation set
    final_predictions = pipeline.best_model.predict(X_val)
    final_rmse = np.sqrt(mean_squared_error(y_val, final_predictions))
    final_r2 = r2_score(y_val, final_predictions)
    
    print("\nFinal Validation Results:")
    print(f"RMSE: {final_rmse:.2f}")
    print(f"R2 Score: {final_r2:.4f}")
    
    # If feature importance analysis was successful, print top features
    if feature_importance is not None:
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
    
    # Make predictions on test set
    if test_df is not None:
        X_test = prepare_features(test_processed)
        predictions = pipeline.best_model.predict(X_test)
        
        submission = pd.DataFrame({
            'id': test_df['id'],
            'monthly_revenue': predictions
        })
        submission.to_csv('submissions/submission_improved.csv', index=False)
        print("\nPredictions saved to submissions/submission_improved.csv")

if __name__ == "__main__":
    main()
