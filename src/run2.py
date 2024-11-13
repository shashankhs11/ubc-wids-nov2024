import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.inspection import partial_dependence
import lightgbm as lgb
from scipy.stats import uniform, randint

def load_data(train_path, test_path=None):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path else None
    return train_df, test_df

def process_price(price_str):
    """
    Process price strings into float values, handling missing values and invalid inputs.
    
    Args:
        price_str: Input price string (can be str, float, int, or None)
        
    Returns:
        float: Processed price value
        None: If the input is missing or invalid
    """
    # Handle None, NaN, or 'missing' values
    if pd.isna(price_str) or price_str == 'missing':
        return None
    
    try:
        # Convert to string if not already
        price_str = str(price_str)
        # Remove currency symbols and commas
        cleaned_price = price_str.replace('$', '').replace(',', '').strip()
        # Convert to float
        return float(cleaned_price)
    except (ValueError, AttributeError):
        # Return None for any conversion errors
        return None

def process_percentage(value):
    """Convert percentage strings to float values."""
    if pd.isna(value) or value == 'missing':
        return None
    try:
        if isinstance(value, str):
            return float(value.strip('%')) / 100.0
        return float(value) / 100.0 if value else None
    except (ValueError, AttributeError):
        return None

def enhanced_feature_engineering(df, is_training=True):
    df_processed = df.copy()
    
    # Basic cleaning
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].fillna('missing')
    
    # Process percentage columns
    percentage_columns = ['host_response_rate', 'host_acceptance_rate']
    for col in percentage_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].apply(process_percentage)
            
    # Advanced price processing
    if 'price' in df_processed.columns:
        df_processed['price'] = df_processed['price'].apply(process_price)
        
    # Enhanced amenities processing
    if 'amenities' in df_processed.columns:
        important_amenities = ['Wifi', 'Kitchen', 'Air conditioning', 'Heating', 
                             'Washer', 'Dryer', 'Pool', 'Free parking']
        for amenity in important_amenities:
            df_processed[f'has_{amenity.lower().replace(" ", "_")}'] = \
                df_processed['amenities'].str.contains(amenity, case=False).astype(int)
        df_processed['amenities_count'] = df_processed['amenities'].str.count(',') + 1
        
    # Location features
    if all(col in df_processed.columns for col in ['latitude', 'longitude']):
        # Calculate distance to center and create location clusters
        center_lat = df_processed['latitude'].mean()
        center_lon = df_processed['longitude'].mean()
        df_processed['distance_to_center'] = np.sqrt(
            (df_processed['latitude'] - center_lat)**2 +
            (df_processed['longitude'] - center_lon)**2
        )
        
    # Enhanced review features
    review_cols = [col for col in df_processed.columns if col.startswith('review_scores_')]
    if review_cols:
        df_processed[review_cols] = df_processed[review_cols].fillna(df_processed[review_cols].median())
        df_processed['avg_review_score'] = df_processed[review_cols].mean(axis=1)
        df_processed['review_score_std'] = df_processed[review_cols].std(axis=1)
        
    # Availability features
    if 'availability_365' in df_processed.columns:
        df_processed['availability_rate'] = df_processed['availability_365'] / 365
        
    # Host features
    if 'host_since' in df_processed.columns:
        df_processed['host_since'] = pd.to_datetime(df_processed['host_since'])
        df_processed['host_years_experience'] = \
            (datetime.now() - df_processed['host_since']).dt.days / 365
            
    # Interaction features
    if all(col in df_processed.columns for col in ['price', 'accommodates']):
        df_processed['price_per_person'] = df_processed['price'] / df_processed['accommodates']
        
    # Categorical encoding with frequency encoding for high-cardinality
    categorical_cols = ['room_type', 'property_type', 'neighbourhood_cleansed']
    encoders = {}
    
    for col in categorical_cols:
        if col in df_processed.columns:
            if is_training:
                freq_encode = df_processed[col].value_counts(normalize=True)
                encoders[f'{col}_freq'] = freq_encode
                df_processed[f'{col}_freq'] = df_processed[col].map(freq_encode)
            # else:
            #     df_processed[f'{col}_freq'] = df_processed[col].map(encoders[f'{col}_freq']).fillna(0)
    return df_processed, encoders if is_training else df_processed

def prepare_features(df):
    # Expanded feature set
    numeric_features = [
        'host_response_rate', 'host_acceptance_rate', 'host_listings_count',
        'accommodates', 'beds', 'price', 'minimum_nights', 'maximum_nights',
        'availability_365', 'number_of_reviews', 'reviews_per_month',
        'avg_review_score', 'review_score_std', 'amenities_count',
        'distance_to_center', 'availability_rate', 'host_years_experience',
        'price_per_person'
    ]
    
    categorical_features = [
        'room_type_freq', 'property_type_freq', 'neighbourhood_cleansed_freq'
    ]
    
    amenity_features = [col for col in df.columns if col.startswith('has_')]
    
    all_features = numeric_features + categorical_features + amenity_features
    selected_features = [col for col in all_features if col in df.columns]
    
    return df[selected_features].fillna(0)

def optimize_and_train_model(X, y):
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Define models and their parameter distributions
    
    models = {
        'xgboost': (xgb.XGBRegressor(random_state=42), {
            'max_depth': randint(3, 12),
            'learning_rate': uniform(0.01, 0.3),
            'n_estimators': randint(100, 1000),
            'min_child_weight': randint(1, 7),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 5),
            'reg_alpha': uniform(0, 5),
            'reg_lambda': uniform(0, 5)
        })
    }
    
    best_model = None
    best_score = float('-inf')
    
    # Number of iterations for random search
    n_iter = 10
    
    for name, (model, param_dist) in models.items():
        print(f"\nOptimizing {name}...")
        
        # RandomizedSearchCV instead of GridSearchCV
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=2,
            random_state=42
        )
        
        random_search.fit(X_train_scaled, y_train)
        
        print(f"\nBest parameters for {name}:")
        for param, value in random_search.best_params_.items():
            print(f"{param}: {value:.4f}" if isinstance(value, float) else f"{param}: {value}")
        print(f"Best cross-validation score: {random_search.best_score_:.4f}")
        
        # Evaluate on validation set
        val_pred = random_search.predict(X_val_scaled)
        r2 = r2_score(y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        print(f"Validation R2 Score: {r2:.4f}")
        print(f"Validation RMSE: {rmse:.2f}")
        
        # Learning curves for the best model
        # plt.figure(figsize=(10, 6))
        # results = pd.DataFrame(random_search.cv_results_)
        # plt.scatter(results['mean_fit_time'], results['mean_test_score'])
        # plt.xlabel('Mean Fit Time (seconds)')
        # plt.ylabel('Mean CV Score')
        # plt.title(f'{name} Performance vs Time')
        # plt.show()
        
        # Plot partial dependence plots for top features
        if r2 > best_score:
            best_score = r2
            best_model = random_search.best_estimator_
            
            # Get feature importances
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Plot partial dependence for top 3 features
            top_features = feature_importance['feature'].head(3).tolist()
            
            # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            # for i, feature in enumerate(top_features):
            #     feature_idx = list(X.columns).index(feature)
            #     pdp = partial_dependence(best_model, X_train_scaled, [feature_idx])
            #     axes[i].plot(pdp[1][0], pdp[0][0])
            #     axes[i].set_xlabel(feature)
            #     axes[i].set_ylabel('Partial dependence')
            # plt.tight_layout()
            # plt.show()
    
    # Feature importance plot
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # plt.figure(figsize=(12, 8))
    # sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
    # plt.title('Top 15 Most Important Features')
    # plt.tight_layout()
    # plt.show()
    
    return best_model, scaler

def main():
    # Load data
    train_df, test_df = load_data('input/train.csv', 'input/test.csv')
    
    # Enhanced feature engineering
    train_processed, encoders = enhanced_feature_engineering(train_df, is_training=True)
    if test_df is not None:
        test_processed = enhanced_feature_engineering(test_df, is_training=False)[0]
    
    # Prepare features
    X = prepare_features(train_processed)
    y = train_processed['monthly_revenue']
    
    print("\nFeatures used in the model:")
    print(X.columns.tolist())
    
    # Train model with optimization
    best_model, scaler = optimize_and_train_model(X, y)
    
    # Make predictions if test data is available
    if test_df is not None:
        X_test = prepare_features(test_processed)
        X_test_scaled = scaler.transform(X_test)
        predictions = best_model.predict(X_test_scaled)
        
        submission = pd.DataFrame({
            'id': test_df['id'],
            'monthly_revenue': predictions
        })
        submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()