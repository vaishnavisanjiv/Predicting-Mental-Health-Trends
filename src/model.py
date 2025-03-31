import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from src.utils import preprocess_data

def train_model():
    """Train the mental health prediction model using XGBoost."""
    print("Loading and preprocessing data...")
    # Load and preprocess data
    df = pd.read_csv("data/mental_health_data.csv")
    df = preprocess_data(df)
    
    # Enhanced feature set
    features = [
        'Age', 'Gender', 'CGPA', 'StudyHoursPerWeek', 
        'AcademicEngagement', 'SleepQuality', 'StudyStressLevel',
        'YearOfStudy'
    ]
    
    # Add course dummy variables
    course_dummies = [col for col in df.columns if col.startswith('Course_')]
    features.extend(course_dummies)
    
    # Add interaction features
    df['Stress_Sleep'] = df['StudyStressLevel'] * df['SleepQuality']
    df['Academic_Stress'] = df['StudyStressLevel'] * df['AcademicEngagement']
    df['Hours_Performance'] = df['StudyHoursPerWeek'] * df['CGPA']
    df['Sleep_Engagement'] = df['SleepQuality'] * df['AcademicEngagement']
    
    features.extend(['Stress_Sleep', 'Academic_Stress', 'Hours_Performance', 'Sleep_Engagement'])
    
    # Target variables
    targets = ['Depression', 'Anxiety', 'PanicAttack']
    
    print("\nTraining XGBoost models...")
    print("-" * 50)
    
    # Split features and targets
    X = df[features]
    y = df[targets]
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost parameters optimized for this task
    xgb_params = {
        'max_depth': 6,
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'random_state': 42
    }
    
    results = {}
    
    for target in targets:
        print(f"\n{target} Model:")
        
        # Create and train XGBoost model
        model = XGBClassifier(**xgb_params)
        model.fit(X_train_scaled, y_train[target])
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test[target], y_pred)
        report = classification_report(y_test[target], y_pred)
        
        print(f"Accuracy: {accuracy:.2%}")
        print("\nDetailed Classification Report:")
        print(report)
        
        # Store results
        results[target] = {
            'accuracy': accuracy,
            'report': report,
            'model': model,
            'scaler': scaler,
            'feature_importance': dict(zip(features, model.feature_importances_)),
            'features': features
        }
        
        # Save models and scalers
        joblib.dump(model, f'models/saved_models/{target}_model.joblib')
        joblib.dump(scaler, f'models/saved_models/{target}_scaler.joblib')
        joblib.dump(features, f'models/saved_models/{target}_features.joblib')
    
    return results

def predict_mental_health(input_data):
    """Make predictions for new input data."""
    predictions = {}
    for target in ['Depression', 'Anxiety', 'PanicAttack']:
        model = joblib.load(f'models/saved_models/{target}_model.joblib')
        scaler = joblib.load(f'models/saved_models/{target}_scaler.joblib')
        features = joblib.load(f'models/saved_models/{target}_features.joblib')
        
        # Add interaction features
        input_data['Stress_Sleep'] = input_data['StudyStressLevel'] * input_data['SleepQuality']
        input_data['Academic_Stress'] = input_data['StudyStressLevel'] * input_data['AcademicEngagement']
        input_data['Hours_Performance'] = input_data['StudyHoursPerWeek'] * input_data['CGPA']
        input_data['Sleep_Engagement'] = input_data['SleepQuality'] * input_data['AcademicEngagement']
        
        # Ensure input data has all required features
        missing_features = set(features) - set(input_data.columns)
        for feature in missing_features:
            input_data[feature] = 0
        
        # Reorder columns to match training data
        input_data = input_data[features]
        
        # Scale input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict_proba(input_scaled)[0]
        predictions[target] = {
            'probability': prediction[1],
            'prediction': 1 if prediction[1] > 0.5 else 0
        }
    
    return predictions

def get_feature_importance():
    """Get feature importance for each target variable."""
    importance_dict = {}
    for target in ['Depression', 'Anxiety', 'PanicAttack']:
        model = joblib.load(f'models/saved_models/{target}_model.joblib')
        features = joblib.load(f'models/saved_models/{target}_features.joblib')
        importance_dict[target] = dict(zip(
            features,
            model.feature_importances_
        ))
    return importance_dict

if __name__ == "__main__":
    print("Starting model training with XGBoost...")
    results = train_model()
