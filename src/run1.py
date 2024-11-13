import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import re

def load_data(train_path, test_path=None):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path else None
    return train_df, test_df

def basic_eda(df):
    print(df.info())
    print(df.describe(include='all'))
    missing_vals = df.isnull().sum().sort_values(ascending=False)
    print(missing_vals[missing_vals > 0])
    plt.figure(figsize=(10, 5))
    sns.histplot(df['monthly_revenue'], bins=50)
    plt.title('Distribution of Monthly Revenue')
    plt.show()
    plt.figure(figsize=(15, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()

def feature_engineering(df):
    tfidf = TfidfVectorizer(max_features=50, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['neighborhood_overview'].fillna(''))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    df = pd.concat([df, tfidf_df], axis=1)
    
    response_time_map = {
        'within an hour': 4, 
        'within a few hours': 3, 
        'within a day': 2, 
        'a few days or more': 1,
        'not_specified': 0
    }
    df['host_response_time'] = df['host_response_time'].fillna('not_specified').map(response_time_map)
    
    for col in ['host_response_rate', 'host_acceptance_rate']:
        df[col] = df[col].replace('%', '', regex=True).astype(float) / 100
    
    df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0}).fillna(0)
    
    df['amenities_count'] = df['amenities'].apply(lambda x: len(re.findall(r"\'(.+?)\'", x)) if pd.notnull(x) else 0)
    
    # Process price column by removing "$" and commas, then converting to float
    if 'price' in df.columns:
        df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    
    categorical_cols = ['room_type', 'property_type', 'neighbourhood_cleansed']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].fillna(''))
        label_encoders[col] = le
    
    center_lat, center_lon = df['latitude'].mean(), df['longitude'].mean()
    df['distance_to_center'] = np.sqrt(
        (df['latitude'] - center_lat)**2 + (df['longitude'] - center_lon)**2
    )
    
    return df, label_encoders

def prepare_features(df, tfidf_df_columns):
    features = [
        'host_response_time', 'host_response_rate', 'host_acceptance_rate',
        'host_is_superhost', 'host_listings_count', 'accommodates', 'beds', 'price',
        'minimum_nights', 'maximum_nights', 'availability_365', 'number_of_reviews',
        'reviews_per_month', 'amenities_count', 'distance_to_center',
        'room_type_encoded', 'property_type_encoded', 'neighbourhood_cleansed_encoded'
    ] + tfidf_df_columns
    return df[features].fillna(0)

def train_model(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    
    for model, name in zip([rf_model, gb_model], ['Random Forest', 'Gradient Boosting']):
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        print(f"{name} RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    return rf_model, gb_model

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

def main(train_path, test_path, submission_path='submission.csv'):
    train_df, test_df = load_data(train_path, test_path)
    # basic_eda(train_df)
    
    train_df, label_encoders = feature_engineering(train_df)
    tfidf_df_columns = [col for col in train_df.columns if col.startswith('neighborhood_overview')]
    X_train = prepare_features(train_df, tfidf_df_columns)
    y_train = train_df['monthly_revenue']
    
    rf_model, gb_model = train_model(X_train, y_train)
    best_rf_model = tune_model(X_train, y_train)
    
    if test_df is not None:
        test_df, _ = feature_engineering(test_df)
        X_test = prepare_features(test_df, tfidf_df_columns)
        X_test_scaled = StandardScaler().fit_transform(X_test)
        
        test_predictions = best_rf_model.predict(X_test_scaled)
        
        submission = pd.DataFrame({
            'id': test_df['id'],
            'monthly_revenue': test_predictions
        })
        submission.to_csv(submission_path, index=False)
        print(f"Submission file saved to {submission_path}")

if __name__ == "__main__":
    main('input/train.csv', 'input/test.csv')
    # main('/mnt/data/small_train.csv', '/mnt/data/test.csv')
