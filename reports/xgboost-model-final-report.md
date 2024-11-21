# XGBoost Regression Model Report: Airbnb Monthly Revenue Prediction

## Model Overview
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Objective**: Predict monthly revenue for Airbnb listings
- **Model Type**: Regression

## Feature Engineering Techniques
1. **Name Feature Extraction**
   - Extracted features from listing names
   - Parsed information about:
     - Number of bedrooms
     - Number of bathrooms
     - Private/shared bath
     - Overall rating
     - New property status

2. **Neighborhood Overview Processing**
   - Cleaned text descriptions
   - Applied TF-IDF vectorization
   - Extracted 100 most important text features

3. **Advanced Feature Engineering**
   - Created revenue-related features
     - Minimum monthly revenue
     - Maximum monthly revenue
     - Revenue per booking
   - Computed availability ratios
   - Calculated distance from city center
   - Generated monthly revenue projections

## Preprocessing Pipeline
- **Numerical Features**
  - Median imputation
  - Standard scaling
  - Includes engineered features and TF-IDF vectors

- **Categorical Features**
  - Constant value imputation
  - One-hot encoding
  - Handles unknown categories

## Model Hyperparameters
```python
XGBRegressor(
    random_state=42,
    colsample_bytree=0.6661,
    learning_rate=0.0147,
    max_depth=3,
    min_child_weight=1,
    n_estimators=235,
    subsample=0.6022
)
```

## Performance Metrics

### Training Performance
- **Mean Squared Error (MSE)**: 1,175,188.32
- **R-squared (R2)**: 0.3541

### Validation Performance
- **Mean Squared Error (MSE)**: 1,180,195.36
- **R-squared (R2)**: 0.2732

## Model Interpretation
- Moderate predictive power with R2 around 0.35
- Indicates significant variability in monthly revenue
- Captures about 35% of the variance in the training data
- Slight overfitting (small difference between train and validation R2)

## Potential Improvements
1. Feature engineering
   - Create more interaction features
   - Explore non-linear transformations
2. Hyperparameter tuning
   - More extensive grid/random search
   - Consider ensemble methods
3. Advanced techniques
   - Feature selection
   - Try other algorithms (Random Forest, Gradient Boosting)

## Key Takeaways
- XGBoost provides a solid baseline for predicting Airbnb monthly revenue
- Complex feature engineering significantly contributes to model performance
- Room for improvement through advanced modeling techniques

## Submission
- Model predictions saved as `submission_xgboost_tuned.csv`
