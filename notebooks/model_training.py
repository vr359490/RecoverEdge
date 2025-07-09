"""
Local Model Training Script
Replaces Databricks notebook with local MLflow tracking
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import pickle
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
LOCAL_MODELS_PATH = Path("./models")
LOCAL_MLRUNS_PATH = Path("./mlruns")
LOCAL_DATA_PATH = Path("./data")

# Create directories
for path in [LOCAL_MODELS_PATH, LOCAL_MLRUNS_PATH, LOCAL_DATA_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Set MLflow tracking to local directory
mlflow.set_tracking_uri(f"file://{LOCAL_MLRUNS_PATH.absolute()}")

class RecoveryRecommendationModel(mlflow.pyfunc.PythonModel):
    """Custom MLflow model for recovery recommendations"""
    
    def __init__(self, effectiveness_model, engagement_model, encoders, scaler):
        self.effectiveness_model = effectiveness_model
        self.engagement_model = engagement_model
        self.encoders = encoders
        self.scaler = scaler
        
        # Recovery methods with metadata
        self.recovery_methods = {
            "Foam Rolling - Legs": {"min_time": 10, "equipment": ["foam_roller"], "difficulty": "medium"},
            "Deep Breathing": {"min_time": 5, "equipment": [], "difficulty": "easy"},
            "Cold Water Immersion": {"min_time": 10, "equipment": ["cold_water"], "difficulty": "hard"},
            "Gentle Stretching": {"min_time": 10, "equipment": ["mat"], "difficulty": "easy"},
            "Massage Gun Therapy": {"min_time": 10, "equipment": ["massage_gun"], "difficulty": "medium"},
            "Progressive Muscle Relaxation": {"min_time": 15, "equipment": [], "difficulty": "easy"},
            "Tennis Ball Foot Massage": {"min_time": 5, "equipment": ["tennis_ball"], "difficulty": "easy"},
            "Legs Up The Wall": {"min_time": 10, "equipment": ["wall"], "difficulty": "easy"}
        }
    
    def predict(self, context, model_input):
        """Generate personalized recommendations"""
        predictions = []
        
        for index, row in model_input.iterrows():
            # Prepare features
            features = self._prepare_features(row)
            
            # Get predictions
            effectiveness_score = float(self.effectiveness_model.predict(features)[0])
            engagement_prob = float(self.engagement_model.predict_proba(features)[0, 1])
            
            # Get context for this prediction
            available_time = row.get('available_time', 15)
            available_equipment = row.get('available_equipment', [])
            fitness_level = row.get('fitness_level', 'beginner')
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                effectiveness_score, 
                engagement_prob,
                available_time,
                available_equipment,
                fitness_level
            )
            
            predictions.append({
                "user_index": index,
                "effectiveness_score": effectiveness_score,
                "engagement_probability": engagement_prob,
                "recommended_methods": recommendations,
                "optimal_duration": available_time
            })
        
        return predictions
    
    def _prepare_features(self, row):
        """Prepare features for prediction"""
        # Create a copy
        features = row.copy()
        
        # Encode categorical variables
        if 'fitness_level' in row and row['fitness_level'] in self.encoders['fitness_level'].classes_:
            features['fitness_level_encoded'] = self.encoders['fitness_level'].transform([row['fitness_level']])[0]
        else:
            features['fitness_level_encoded'] = self.encoders['fitness_level'].transform(['beginner'])[0]
        
        if 'age_group' in row and row['age_group'] in self.encoders['age_group'].classes_:
            features['age_group_encoded'] = self.encoders['age_group'].transform([row['age_group']])[0]
        else:
            features['age_group_encoded'] = self.encoders['age_group'].transform(['unknown'])[0]
        
        # Select and order features
        feature_cols = [
            'total_interactions', 'unique_sessions', 'avg_response_length',
            'total_sessions', 'avg_session_duration', 'avg_satisfaction',
            'avg_methods_completed', 'completion_efficiency',
            'avg_method_effectiveness', 'fitness_level_encoded', 'age_group_encoded',
            'preferred_locations_count', 'equipment_variety'
        ]
        
        # Ensure all features exist
        for col in feature_cols:
            if col not in features:
                features[col] = 0
        
        # Scale features
        feature_array = features[feature_cols].values.reshape(1, -1)
        feature_array = self.scaler.transform(feature_array)
        
        return feature_array
    
    def _generate_recommendations(self, effectiveness_score, engagement_prob, 
                                 available_time, available_equipment, fitness_level):
        """Generate method recommendations based on scores and context"""
        recommendations = []
        
        # Score each method
        method_scores = {}
        for method, info in self.recovery_methods.items():
            score = self._calculate_method_score(
                method, info, effectiveness_score, engagement_prob,
                available_time, available_equipment, fitness_level
            )
            method_scores[method] = score
        
        # Sort by score and take top 3
        sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for method, score in sorted_methods:
            confidence = min(score / 100, 0.95)  # Normalize to confidence
            reason = self._get_recommendation_reason(
                method, effectiveness_score, engagement_prob, fitness_level
            )
            
            recommendations.append({
                "method": method,
                "confidence": round(confidence, 2),
                "reason": reason
            })
        
        return recommendations
    
    def _calculate_method_score(self, method, info, effectiveness_score, 
                               engagement_prob, available_time, available_equipment, fitness_level):
        """Calculate score for a specific method"""
        score = 50  # Base score
        
        # Effectiveness component (0-30 points)
        if effectiveness_score >= 4.0:
            if info['difficulty'] in ['medium', 'hard']:
                score += 30
            else:
                score += 20
        elif effectiveness_score >= 3.5:
            if info['difficulty'] in ['easy', 'medium']:
                score += 25
            else:
                score += 15
        else:
            if info['difficulty'] == 'easy':
                score += 25
            else:
                score += 10
        
        # Engagement component (0-20 points)
        score += engagement_prob * 20
        
        # Time availability (0-20 points)
        if available_time >= info['min_time']:
            score += 20
        else:
            score += max(0, 20 - (info['min_time'] - available_time) * 2)
        
        # Equipment availability (0-15 points)
        if not info['equipment']:  # No equipment needed
            score += 15
        elif all(eq in available_equipment for eq in info['equipment']):
            score += 15
        elif any(eq in available_equipment for eq in info['equipment']):
            score += 7
        
        # Fitness level match (0-15 points)
        if fitness_level == 'beginner' and info['difficulty'] == 'easy':
            score += 15
        elif fitness_level == 'intermediate' and info['difficulty'] == 'medium':
            score += 15
        elif fitness_level == 'advanced' and info['difficulty'] == 'hard':
            score += 15
        else:
            score += 5
        
        return score
    
    def _get_recommendation_reason(self, method, effectiveness_score, engagement_prob, fitness_level):
        """Generate reason for recommendation"""
        if method == "Deep Breathing":
            return "Universal benefit, no equipment needed"
        elif method == "Foam Rolling - Legs":
            if effectiveness_score >= 4.0:
                return "Proven effective for your profile"
            else:
                return "Great for muscle recovery"
        elif method == "Cold Water Immersion":
            return "Advanced recovery technique"
        elif method == "Gentle Stretching":
            if fitness_level == 'beginner':
                return "Perfect for your fitness level"
            else:
                return "Maintains flexibility"
        elif method == "Legs Up The Wall":
            return "Passive recovery, easy to perform"
        else:
            return "Recommended based on your preferences"

def load_or_create_training_data():
    """Load training data from features or create synthetic data"""
    features_path = LOCAL_DATA_PATH / "user_features.csv"
    
    if features_path.exists():
        logger.info("Loading features from CSV...")
        df = pd.read_csv(features_path)
        logger.info(f"Loaded {len(df)} user features")
        return df
    else:
        logger.info("Creating synthetic training data...")
        return create_synthetic_data()

def create_synthetic_data(n_samples=1000):
    """Create synthetic training data"""
    np.random.seed(42)
    
    # Create base features
    data = {
        'user_id': [f'user_{i}' for i in range(n_samples)],
        'total_interactions': np.random.poisson(10, n_samples),
        'unique_sessions': np.random.poisson(5, n_samples),
        'avg_response_length': np.random.normal(100, 20, n_samples),
        'total_sessions': np.random.poisson(8, n_samples),
        'avg_session_duration': np.random.normal(15, 5, n_samples),
        'avg_satisfaction': np.random.uniform(2, 5, n_samples),
        'avg_methods_completed': np.random.uniform(1, 5, n_samples),
        'completion_efficiency': np.random.uniform(0.3, 1.0, n_samples),
        'avg_method_effectiveness': np.random.uniform(2, 5, n_samples),
        'fitness_level': np.random.choice(['beginner', 'intermediate', 'advanced'], n_samples, p=[0.5, 0.35, 0.15]),
        'age_group': np.random.choice(['18-25', '26-35', '36-45', '46+'], n_samples, p=[0.25, 0.35, 0.25, 0.15]),
        'preferred_locations_count': np.random.randint(1, 4, n_samples),
        'equipment_variety': np.random.randint(0, 6, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlations to make it more realistic
    # Higher satisfaction correlates with higher effectiveness
    df.loc[df['avg_satisfaction'] > 4, 'avg_method_effectiveness'] += np.random.uniform(0.5, 1.0, sum(df['avg_satisfaction'] > 4))
    
    # More sessions correlate with better completion efficiency
    df.loc[df['total_sessions'] > 10, 'completion_efficiency'] += np.random.uniform(0.1, 0.2, sum(df['total_sessions'] > 10))
    
    # Clip values to reasonable ranges
    df['avg_method_effectiveness'] = df['avg_method_effectiveness'].clip(1, 5)
    df['completion_efficiency'] = df['completion_efficiency'].clip(0, 1)
    
    return df

def train_models(df):
    """Train all models with MLflow tracking"""
    mlflow.set_experiment("local_recovery_recommendations")
    
    # Prepare features
    logger.info("Preparing features...")
    
    # Encode categorical variables
    le_fitness = LabelEncoder()
    le_age = LabelEncoder()
    
    df['fitness_level_encoded'] = le_fitness.fit_transform(df['fitness_level'])
    df['age_group_encoded'] = le_age.fit_transform(df['age_group'])
    
    # Select feature columns
    feature_cols = [
        'total_interactions', 'unique_sessions', 'avg_response_length',
        'total_sessions', 'avg_session_duration', 'avg_satisfaction',
        'avg_methods_completed', 'completion_efficiency',
        'avg_method_effectiveness', 'fitness_level_encoded', 'age_group_encoded',
        'preferred_locations_count', 'equipment_variety'
    ]
    
    X = df[feature_cols]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Train Model 1: Effectiveness Predictor
    logger.info("Training effectiveness predictor...")
    with mlflow.start_run(run_name="effectiveness_predictor"):
        y_effectiveness = df['avg_method_effectiveness']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_effectiveness, test_size=0.2, random_state=42
        )
        
        # Train model
        effectiveness_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        effectiveness_model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = effectiveness_model.predict(X_train)
        test_pred = effectiveness_model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(effectiveness_model, X_scaled, y_effectiveness, cv=5, scoring='r2')
        
        # Log metrics
        mlflow.log_params({
            "model_type": "GradientBoostingRegressor",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6
        })
        
        mlflow.log_metrics({
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "cv_mean_r2": cv_scores.mean(),
            "cv_std_r2": cv_scores.std()
        })
        
        # Log model
        mlflow.sklearn.log_model(effectiveness_model, "model")
        
        # Save locally
        with open(LOCAL_MODELS_PATH / "effectiveness_model.pkl", 'wb') as f:
            pickle.dump(effectiveness_model, f)
        
        logger.info(f"Effectiveness Model - Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
    
    # Train Model 2: Engagement Predictor
    logger.info("Training engagement predictor...")
    with mlflow.start_run(run_name="engagement_predictor"):
        y_engagement = (df['completion_efficiency'] >= 0.8).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_engagement, test_size=0.2, random_state=42, stratify=y_engagement
        )
        
        # Train model
        engagement_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        
        engagement_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = engagement_model.predict(X_test)
        y_pred_proba = engagement_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': engagement_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Log metrics
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": 10,
            "class_weight": "balanced"
        })
        
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        })
        
        # Log feature importance
        mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")
        
        # Log model
        mlflow.sklearn.log_model(engagement_model, "model")
        
        # Save locally
        with open(LOCAL_MODELS_PATH / "engagement_model.pkl", 'wb') as f:
            pickle.dump(engagement_model, f)
        
        logger.info(f"Engagement Model - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
    
    # Train Combined Model
    logger.info("Creating combined recommendation model...")
    with mlflow.start_run(run_name="combined_recommendation_model"):
        # Create combined model
        combined_model = RecoveryRecommendationModel(
            effectiveness_model=effectiveness_model,
            engagement_model=engagement_model,
            encoders={'fitness_level': le_fitness, 'age_group': le_age},
            scaler=scaler
        )
        
        # Save artifacts
        artifacts = {
            'encoders': {'fitness_level': le_fitness, 'age_group': le_age},
            'scaler': scaler
        }
        
        with open(LOCAL_MODELS_PATH / "model_artifacts.pkl", 'wb') as f:
            pickle.dump(artifacts, f)
        
        # Log combined model
        mlflow.pyfunc.log_model(
            "model",
            python_model=combined_model,
            artifacts={
                "effectiveness_model": str(LOCAL_MODELS_PATH / "effectiveness_model.pkl"),
                "engagement_model": str(LOCAL_MODELS_PATH / "engagement_model.pkl"),
                "artifacts": str(LOCAL_MODELS_PATH / "model_artifacts.pkl")
            }
        )
        
        # Test the combined model
        test_input = pd.DataFrame([{
            'user_id': 'test_user',
            'total_interactions': 10,
            'unique_sessions': 5,
            'avg_response_length': 100,
            'total_sessions': 8,
            'avg_session_duration': 15,
            'avg_satisfaction': 4.0,
            'avg_methods_completed': 3,
            'completion_efficiency': 0.8,
            'avg_method_effectiveness': 4.0,
            'fitness_level': 'intermediate',
            'age_group': '26-35',
            'preferred_locations_count': 2,
            'equipment_variety': 3,
            'available_time': 20,
            'available_equipment': ['foam_roller', 'mat']
        }])
        
        predictions = combined_model.predict(None, test_input)
        
        mlflow.log_text(str(predictions[0]), "sample_prediction.txt")
        
        logger.info("Combined model created and logged")
        logger.info(f"Sample prediction: {predictions[0]}")
    
    return effectiveness_model, engagement_model, artifacts

from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    logger.info("Starting model training pipeline...")
    
    # Load or create data
    df = load_or_create_training_data()
    
    # Save training data
    df.to_csv(LOCAL_DATA_PATH / "training_data.csv", index=False)
    logger.info(f"Training data saved to {LOCAL_DATA_PATH / 'training_data.csv'}")
    
    # Train models
    effectiveness_model, engagement_model, artifacts = train_models(df)
    
    logger.info("\nModel training completed!")
    logger.info(f"Models saved to: {LOCAL_MODELS_PATH}")
    logger.info(f"MLflow runs saved to: {LOCAL_MLRUNS_PATH}")
    logger.info("\nTo view MLflow UI, run: mlflow ui --backend-store-uri file://./mlruns")