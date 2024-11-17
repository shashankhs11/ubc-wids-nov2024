import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Feature extraction functions
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

# Load and preprocess data
def preprocess_data(train_df, test_df=None):
    # Define features to drop
    drop_features = ['host_id', 'host_name', 'latitude', 'longitude', 'amenities', 
                    'bathrooms', 'bedrooms', 'host_listings_count', 
                    'host_total_listings_count', 'neighbourhood', 'name']
    
    # Create copies to avoid modifying original dataframes
    train_df = train_df.copy()
    test_df = test_df.copy() if test_df is not None else None
    
    # Store name column before dropping for feature extraction
    train_name = train_df['name']
    test_name = test_df['name'] if test_df is not None else None
    
    # Drop unnecessary features
    for df in [train_df] + ([test_df] if test_df is not None else []):
        # Check which features exist in the dataframe before dropping
        features_to_drop = [f for f in drop_features if f in df.columns]
        df.drop(columns=features_to_drop, inplace=True)
    
    # Process name column features
    for df, name_series in [(train_df, train_name)] + ([(test_df, test_name)] if test_df is not None else []):
        df["split_parts"] = name_series.str.split("·")
        df["bedrooms"] = df["split_parts"].apply(extract_bedrooms)
        df["beds"] = df["split_parts"].apply(extract_beds)
        df["baths"] = df["split_parts"].apply(extract_baths)
        df["is_bath_private"] = df["split_parts"].apply(is_private_bath).astype(int)
        df["is_bath_shared"] = df["split_parts"].apply(is_shared_bath).astype(int)
        df.drop('split_parts', axis=1, inplace=True)
        
        # Convert 'f' and 't' to 0 and 1
        if 'host_is_superhost' in df.columns:
            df['host_is_superhost'] = (df['host_is_superhost'] == 't').astype(int)
        if 'instant_bookable' in df.columns:
            df['instant_bookable'] = (df['instant_bookable'] == 't').astype(int)
    
    # Clean 'price' column before passing to pipeline
    train_df['price'] = train_df['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    test_df['price'] = test_df['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)


    return train_df, test_df

# BERT embedding extraction
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
                    # Handle missing text with zeros
                    embeddings.append(torch.zeros(768))
                    continue
                    
                inputs = self.tokenizer(text, 
                                      return_tensors="pt",
                                      max_length=max_length,
                                      padding=True,
                                      truncation=True)
                
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                embeddings.append(embedding)
                
        return torch.stack(embeddings)

# Custom Dataset
class AirbnbDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]

# Neural Network Model
class RevenuePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Main processing pipeline
def create_feature_pipeline(train_df, test_df=None):
    # Preprocess data
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
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Get BERT embeddings for neighborhood_overview
    bert_embedder = BERTEmbedding()
    neighborhood_overview_train = train_df.get('neighborhood_overview', pd.Series([''] * len(train_df)))
    train_embeddings = bert_embedder.get_embeddings(neighborhood_overview_train)
    
    if test_df is not None:
        neighborhood_overview_test = test_df.get('neighborhood_overview', pd.Series([''] * len(test_df)))
        test_embeddings = bert_embedder.get_embeddings(neighborhood_overview_test)
    
    # Drop neighborhood_overview after getting embeddings
    if 'neighborhood_overview' in train_df.columns:
        train_df.drop('neighborhood_overview', axis=1, inplace=True)
    if test_df is not None and 'neighborhood_overview' in test_df.columns:
        test_df.drop('neighborhood_overview', axis=1, inplace=True)
    
    # Handle missing values in numerical and categorical columns
    # For numerical features, use SimpleImputer to fill NaN values with the mean
    num_imputer = SimpleImputer(strategy='mean')
    
    # For categorical features, use SimpleImputer to fill NaN values with the most frequent value (mode)
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    # Create transformers for numerical and categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', cat_imputer),  # Impute missing categorical values
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', num_imputer),  # Impute missing numerical values
        ('scaler', StandardScaler())  # Scale numerical values
    ])

    # Create preprocessor for categorical and numerical features
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    
    # Filter features to only include those present in the dataframe
    available_cat_features = [f for f in cat_features if f in train_df.columns]
    available_num_features = [f for f in num_features if f in train_df.columns]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, available_cat_features),
            ('num', numerical_transformer, available_num_features)
        ])
    
    # Fit and transform features
    train_features = preprocessor.fit_transform(train_df)
    if test_df is not None:
        test_features = preprocessor.transform(test_df)
    
    # Combine all features
    train_combined = np.hstack([train_features, train_embeddings.numpy()])
    if test_df is not None:
        test_combined = np.hstack([test_features, test_embeddings.numpy()])
    else:
        test_combined = None
    
    return train_combined, test_combined, preprocessor

# Training function
def train_model(train_features, train_targets, val_split=0.2, batch_size=32, epochs=10):
    # Split into train and validation
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    val_size = int(len(train_features) * val_split)
    indices = torch.randperm(len(train_features))
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_dataset = AirbnbDataset(
        torch.FloatTensor(train_features[train_indices]).to(device), 
        torch.FloatTensor(train_targets[train_indices]).to(device)
    )
    val_dataset = AirbnbDataset(
        torch.FloatTensor(train_features[val_indices]).to(device), 
        torch.FloatTensor(train_targets[val_indices]).to(device)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = train_features.shape[1]
    model = RevenuePredictor(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                val_loss += criterion(outputs, targets.unsqueeze(1)).item()
                
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        
    return model

# Main execution
if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv('input/train.csv')
    test_df = pd.read_csv('input/test.csv')
    # train_df.drop(['bathrooms', 'bedrooms'], axis=1, inplace=True)
    train_df.set_index('id', inplace=True)
    test_df.set_index('id', inplace=True)
    
    # Create features
    train_features, test_features, preprocessor = create_feature_pipeline(train_df, test_df)
    
    # Train model
    model = train_model(
        train_features,
        train_df['monthly_revenue'].values,
        val_split=0.2,
        batch_size=32,
        epochs=100
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Make predictions on test set
    model.eval()
    with torch.no_grad():
        test_predictions = model(torch.FloatTensor(test_features).to(device))
        
    # Create submission
    submission = pd.DataFrame({
        'id': test_df.index,
        'monthly_revenue': test_predictions.cpu().numpy().squeeze()
    })
    submission.to_csv('submission.csv', index=False)