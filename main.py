import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os

# Load data
print("Loading data...")
games_df = pd.read_csv('data/games.csv')
players_df = pd.read_csv('data/players.csv')
teams_df = pd.read_csv('data/teams.csv')
ranking_df = pd.read_csv('data/ranking.csv')
games_details_df = pd.read_csv('data/games_details.csv')

print(f"Games data shape: {games_df.shape}")
print(f"Players data shape: {players_df.shape}")
print(f"Teams data shape: {teams_df.shape}")
print(f"Ranking data shape: {ranking_df.shape}")
print(f"Games details data shape: {games_details_df.shape}")

# Prepare features and target using the games dataset
# Select features (excluding game metadata and target variable)
feature_cols = [
    'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home',
    'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away'
]

# Check available features
available_features = [col for col in feature_cols if col in games_df.columns]
print(f"\nUsing features: {available_features}")

X = games_df[available_features].copy()
y = games_df['HOME_TEAM_WINS'].copy()

# Handle missing values
print(f"\nMissing values before cleaning:\n{X.isnull().sum()}")
X = X.dropna()
y = y[X.index]  # Align target with features

print(f"\nData shape after cleaning: X={X.shape}, y={y.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Split data into 80/20 train-test split
print("\nSplitting data into 80% train and 20% test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train Random Forest model
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)
print("Model training complete!")

# Make predictions
print("\nEvaluating model...")
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

# Print evaluation results
print(f"\n{'='*50}")
print("MODEL EVALUATION RESULTS")
print(f"{'='*50}")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_test)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_test)}")

# Feature importance
print(f"\n{'='*50}")
print("FEATURE IMPORTANCE")
print(f"{'='*50}")
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.to_string(index=False))

# Save the model
model_path = 'random_forest_model.pkl'
print(f"\nSaving model to {model_path}...")
joblib.dump(rf_model, model_path)
print("Model saved successfully!")

# Save feature names for later use
features_path = 'model_features.pkl'
joblib.dump(available_features, features_path)
print(f"Feature names saved to {features_path}")
