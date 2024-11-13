import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_data(train_path, test_path=None):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path else None
    return train_df, test_df

def basic_eda(df):
    print("\nBasic Dataset Info:")
    print(f"Shape: {df.shape}")
    print("\nColumns in dataset:")
    print(df.columns.tolist())
    print("\nMissing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print("\nNumeric Columns Statistics:")
    print(df[numeric_cols].describe())
    
    # Create visualizations
    # plt.figure(figsize=(12, 6))
    # sns.histplot(data=df, x='monthly_revenue', bins=50)
    # plt.title('Distribution of Monthly Revenue')
    # plt.show()
    
    # # Correlation matrix for numeric columns
    # plt.figure(figsize=(15, 10))
    # sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
    # plt.title('Correlation Matrix')
    # plt.show()

def safe_process_column(df, column_name, default_value, processor_func):
    """Safely process a column with a fallback to default value if the column doesn't exist"""
    if column_name in df.columns:
        return processor_func(df[column_name])
    return pd.Series([default_value] * len(df))

def process_percentage(x):
    """Convert percentage string to float, handling missing values"""
    if pd.isna(x):
        return 0.0
    if isinstance(x, str):
        return float(x.strip('%')) / 100
    return float(x)

def feature_engineering(df, is_training=True):
    # Create copy to avoid modifying original dataframe
    df_processed = df.copy()
    
    # Handle missing values
    if 'neighborhood_overview' in df_processed.columns:
        df_processed['neighborhood_overview'] = df_processed['neighborhood_overview'].fillna('')
    
    # Process response time and rates
    if 'host_response_time' in df_processed.columns:
        df_processed['host_response_time'] = df_processed['host_response_time'].fillna('not_specified')
    
    # Process percentage columns
    for col in ['host_response_rate', 'host_acceptance_rate']:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].apply(process_percentage)
    
    # Convert boolean columns
    boolean_columns = ['host_is_superhost']
    for col in boolean_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map({'t': 1, 'f': 0, True: 1, False: 0}).fillna(0)
    
    # Process amenities
    if 'amenities' in df_processed.columns:
        df_processed['amenities_count'] = df_processed['amenities'].str.len()
    else:
        df_processed['amenities_count'] = 0
    
    # Price processing
    if 'price' in df_processed.columns:
        df_processed['price'] = df_processed['price'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Create features from review scores
    review_cols = [col for col in df_processed.columns if col.startswith('review_scores_')]
    if review_cols:
        df_processed[review_cols] = df_processed[review_cols].fillna(df_processed[review_cols].mean())
        df_processed['avg_review_score'] = df_processed[review_cols].mean(axis=1)
    else:
        df_processed['avg_review_score'] = 0
    
    # Create categorical encodings
    categorical_cols = ['room_type', 'property_type', 'neighbourhood_cleansed']
    categorical_cols = [col for col in categorical_cols if col in df_processed.columns]
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
        if is_training:
            label_encoders[col] = le
    
    # Location features
    if 'latitude' in df_processed.columns and 'longitude' in df_processed.columns:
        df_processed['distance_to_center'] = np.sqrt(
            (df_processed['latitude'] - df_processed['latitude'].mean())**2 +
            (df_processed['longitude'] - df_processed['longitude'].mean())**2
        )
    else:
        df_processed['distance_to_center'] = 0
    
    return df_processed, label_encoders if is_training else df_processed

def prepare_features(df):
    # Define all possible feature columns
    potential_features = [
        'host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
        'host_listings_count', 'accommodates', 'beds', 'price',
        'minimum_nights', 'maximum_nights', 'availability_365',
        'number_of_reviews', 'reviews_per_month', 'avg_review_score',
        'amenities_count', 'distance_to_center', 'room_type_encoded',
        'property_type_encoded', 'neighbourhood_cleansed_encoded'
    ]
    
    # Only use features that exist in the dataframe
    feature_cols = [col for col in potential_features if col in df.columns]
    
    # Fill any missing values with 0
    return df[feature_cols].fillna(0)

def train_model(X, y):
    # Initialize models
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    # Train and evaluate models
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model
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
    
    # Feature importance for best model
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # plt.figure(figsize=(10, 6))
    # sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    # plt.title('Top 10 Most Important Features')
    # plt.show()
    
    return best_model

def make_predictions(model, X_test):
    return model.predict(X_test)

def tune_model(X, y):
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf_model = RandomForestRegressor(random_state=42)
    rf_random = RandomizedSearchCV(
        rf_model, param_grid, n_iter=10, cv=5, random_state=42, scoring='neg_mean_squared_error'
    )
    rf_random.fit(X, y)
    return rf_random.best_estimator_

def main():
    # Load data
    train_df, test_df = load_data('input/train.csv', 'input/test.csv')
    
    # Perform EDA
    basic_eda(train_df)
    
    # Feature engineering
    train_processed, label_encoders = feature_engineering(train_df, is_training=True)
    if test_df is not None:
        test_processed, label_encoders = feature_engineering(test_df, is_training=False)
    
    # Prepare features
    X = prepare_features(train_processed)
    y = train_processed['monthly_revenue']
    
    print("\nFeatures used in the model:")
    print(X.columns.tolist())
    
    # Train model
    best_model = train_model(X, y)
    best_rf_model = tune_model(X, y)
    
    # Make predictions if test data is available
    if test_df is not None:
        X_test = prepare_features(test_processed)
        predictions = make_predictions(best_rf_model, X_test)
        
        # Create submission file
        submission = pd.DataFrame({
            'id': test_df['id'],
            'monthly_revenue': predictions
        })
        submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()


    # main('input/train.csv', 'input/test.csv')