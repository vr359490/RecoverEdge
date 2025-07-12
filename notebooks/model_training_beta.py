"""
Improved Local Model Training Script
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, mean_absolute_error, roc_auc_score
import pickle
from pathlib import Path
import logging
import warnings
import yaml
from typing import Dict, Any, Tuple
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

class ModelConfig:
    """Configuration management for training"""
    
    def __init__(self, config_path: str = "training_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        default_config = {
            "data": {
                "n_samples": 1000,
                "test_size": 0.2,
                "random_state": 42,
                "cv_folds": 5
            },
            "effectiveness_model": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42
            },
            "engagement_model": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "random_state": 42,
                "class_weight": "balanced"
            }
        }
        
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Could not load config file {self.config_path}: {e}. Using defaults.")
        
        return default_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

class SyntheticDataGenerator:
    """Generates realistic synthetic training data"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.random_state = config.get('data.random_state', 42)
    
    def create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic training data with realistic correlations"""
        n_samples = self.config.get('data.n_samples', 1000)
        np.random.seed(self.random_state)
        
        # Create base user characteristics
        data = {}
        
        # Demographics
        data['user_id'] = [f'user_{i}' for i in range(n_samples)]
        data['fitness_level'] = np.random.choice(['beginner', 'intermediate', 'advanced'], n_samples, p=[0.5, 0.35, 0.15])
        data['age_group'] = np.random.choice(['18-25', '26-35', '36-45', '46+'], n_samples, p=[0.25, 0.35, 0.25, 0.15])
        
        # Create fitness level influence multiplier
        fitness_multiplier = {'beginner': 0.8, 'intermediate': 1.0, 'advanced': 1.3}
        fitness_mult = np.array([fitness_multiplier[level] for level in data['fitness_level']])
        
        # Base engagement factor (drives many correlated behaviors)
        engagement_factor = np.random.beta(2, 2, n_samples)  # More realistic distribution
        
        # User interaction metrics (correlated with engagement and fitness)
        base_sessions = np.random.poisson(5, n_samples)
        data['unique_sessions'] = np.maximum(1, (base_sessions * fitness_mult * (0.5 + engagement_factor)).astype(int))
        data['total_sessions'] = data['unique_sessions'] + np.random.poisson(2, n_samples)
        data['total_interactions'] = data['total_sessions'] * np.random.poisson(3, n_samples)
        
        # Session behavior (more engaged users have longer, more efficient sessions)
        data['avg_session_duration'] = np.maximum(5, 
            np.random.normal(15, 5, n_samples) * (0.7 + 0.6 * engagement_factor)
        )
        data['completion_efficiency'] = np.maximum(0.1, np.minimum(1.0,
            engagement_factor * 0.8 + np.random.normal(0, 0.15, n_samples)
        ))
        
        # Methods completed correlates with engagement and fitness level
        data['avg_methods_completed'] = np.maximum(1, np.minimum(5,
            2 + engagement_factor * 3 * fitness_mult + np.random.normal(0, 0.5, n_samples)
        ))
        
        # Satisfaction correlates with completion efficiency and fitness level
        data['avg_satisfaction'] = np.maximum(1, np.minimum(5,
            2.5 + engagement_factor * 2 + fitness_mult * 0.5 + np.random.normal(0, 0.3, n_samples)
        ))
        
        # Response length (for chat interactions)
        data['avg_response_length'] = np.maximum(20,
            np.random.normal(100, 20, n_samples) * (0.8 + 0.4 * engagement_factor)
        )
        
        # Location and equipment variety
        data['preferred_locations_count'] = np.random.randint(1, 4, n_samples)
        data['equipment_variety'] = np.maximum(0, 
            np.random.poisson(3, n_samples) * fitness_mult
        ).astype(int)
        
        return pd.DataFrame(data)
    
    def create_target_variables(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Create target variables"""
        
        # Effectiveness: based on user characteristics and behaviors (NOT using avg_method_effectiveness)
        fitness_base = {'beginner': 3.0, 'intermediate': 3.5, 'advanced': 4.0}
        base_effectiveness = np.array([fitness_base[level] for level in df['fitness_level']])
        
        # Add realistic correlations
        effectiveness = (
            base_effectiveness + 
            (df['avg_satisfaction'] - 3) * 0.3 +  # Satisfied users see better results
            (df['completion_efficiency'] - 0.5) * 0.5 +  # Consistent users see better results
            np.random.normal(0, 0.4, len(df))  # Natural variation
        )
        effectiveness = np.maximum(1, np.minimum(5, effectiveness))
        
        # Engagement: based on completion efficiency threshold
        engagement = (df['completion_efficiency'] >= 0.8).astype(int)
        
        return pd.Series(effectiveness, name='effectiveness'), pd.Series(engagement, name='high_engagement')

class ModelTrainer:
    """Handles model training with comprehensive evaluation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.encoders = {}
        self.scaler = None
    
    def prepare_features(self, df: pd.DataFrame, fit_transformers: bool = True) -> pd.DataFrame:
        """Prepare features with proper preprocessing"""
        X = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['fitness_level', 'age_group']
        
        for col in categorical_cols:
            if fit_transformers:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                X[f'{col}_encoded'] = self.encoders[col].fit_transform(X[col])
            else:
                if col in self.encoders:
                    try:
                        X[f'{col}_encoded'] = self.encoders[col].transform(X[col])
                    except ValueError:
                        # Handle unknown categories
                        logger.warning(f"Unknown category in {col}, using default encoding")
                        X[f'{col}_encoded'] = 0
        
        # Select numerical features
        feature_cols = [
            'total_interactions', 'unique_sessions', 'avg_response_length',
            'total_sessions', 'avg_session_duration', 'avg_satisfaction',
            'avg_methods_completed', 'completion_efficiency',
            'fitness_level_encoded', 'age_group_encoded',
            'preferred_locations_count', 'equipment_variety'
        ]
        
        X_features = X[feature_cols]
        
        # Scale features
        if fit_transformers:
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_features),
                columns=feature_cols,
                index=X_features.index
            )
        else:
            if self.scaler:
                X_scaled = pd.DataFrame(
                    self.scaler.transform(X_features),
                    columns=feature_cols,
                    index=X_features.index
                )
            else:
                X_scaled = X_features
        
        return X_scaled
    
    def train_effectiveness_model(self, X: pd.DataFrame, y: pd.Series) -> GradientBoostingRegressor:
        """Train effectiveness prediction model with comprehensive evaluation"""
        config = self.config.get('effectiveness_model', {})
        
        model = GradientBoostingRegressor(**config)
        
        # Split data
        test_size = self.config.get('data.test_size', 0.2)
        random_state = self.config.get('data.random_state', 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Comprehensive evaluation
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Cross-validation
        cv_folds = self.config.get('data.cv_folds', 5)
        cv_scores_r2 = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
        cv_scores_rmse = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
        
        # Calculate metrics
        metrics = {
            'effectiveness_train_r2': model.score(X_train, y_train),
            'effectiveness_test_r2': model.score(X_test, y_test),
            'effectiveness_train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'effectiveness_test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'effectiveness_train_mae': mean_absolute_error(y_train, train_pred),
            'effectiveness_test_mae': mean_absolute_error(y_test, test_pred),
            'effectiveness_cv_r2_mean': cv_scores_r2.mean(),
            'effectiveness_cv_r2_std': cv_scores_r2.std(),
            'effectiveness_cv_rmse_mean': np.sqrt(-cv_scores_rmse.mean()),
            'effectiveness_cv_rmse_std': np.sqrt(cv_scores_rmse.std())
        }
        
        # Log parameters and metrics
        mlflow.log_params({f"effectiveness_{k}": v for k, v in config.items()})
        mlflow.log_metrics(metrics)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mlflow.log_text(feature_importance.to_string(), "effectiveness_feature_importance.txt")
        
        logger.info(f"Effectiveness model - Test R²: {metrics['effectiveness_test_r2']:.3f}, "
                   f"Test RMSE: {metrics['effectiveness_test_rmse']:.3f}")
        
        return model
    
    def train_engagement_model(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """Train engagement prediction model with comprehensive evaluation"""
        config = self.config.get('engagement_model', {})
        
        model = RandomForestClassifier(**config)
        
        # Split data
        test_size = self.config.get('data.test_size', 0.2)
        random_state = self.config.get('data.random_state', 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Comprehensive evaluation
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        test_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Cross-validation
        cv_folds = self.config.get('data.cv_folds', 5)
        cv_scores_acc = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        cv_scores_f1 = cross_val_score(model, X, y, cv=cv_folds, scoring='f1')
        
        # Calculate metrics
        metrics = {
            'engagement_train_accuracy': accuracy_score(y_train, train_pred),
            'engagement_test_accuracy': accuracy_score(y_test, test_pred),
            'engagement_test_precision': precision_score(y_test, test_pred, zero_division=0),
            'engagement_test_recall': recall_score(y_test, test_pred, zero_division=0),
            'engagement_test_f1': f1_score(y_test, test_pred, zero_division=0),
            'engagement_cv_accuracy_mean': cv_scores_acc.mean(),
            'engagement_cv_accuracy_std': cv_scores_acc.std(),
            'engagement_cv_f1_mean': cv_scores_f1.mean(),
            'engagement_cv_f1_std': cv_scores_f1.std()
        }
        
        # Add AUC-ROC if we have both classes
        if len(np.unique(y_test)) > 1:
            metrics['engagement_test_auc_roc'] = roc_auc_score(y_test, test_pred_proba)
        
        # Log parameters and metrics
        mlflow.log_params({f"engagement_{k}": v for k, v in config.items()})
        mlflow.log_metrics(metrics)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mlflow.log_text(feature_importance.to_string(), "engagement_feature_importance.txt")
        
        logger.info(f"Engagement model - Test Accuracy: {metrics['engagement_test_accuracy']:.3f}, "
                   f"Test F1: {metrics['engagement_test_f1']:.3f}")
        
        return model

class RecoveryRecommendationModel(mlflow.pyfunc.PythonModel):
    """Enhanced MLflow model for recovery recommendations"""
    
    def __init__(self, effectiveness_model, engagement_model, encoders, scaler):
        self.effectiveness_model = effectiveness_model
        self.engagement_model = engagement_model
        self.encoders = encoders
        self.scaler = scaler
        
        # Recovery methods with realistic metadata
        self.recovery_methods = {
            # Original Methods
            "Foam Rolling - Legs": {"min_time": 8, "equipment": ["foam_roller"], "difficulty": "medium", "benefits": ["circulation", "flexibility"]},
            "Deep Breathing": {"min_time": 5, "equipment": [], "difficulty": "easy", "benefits": ["stress_relief", "recovery"]},
            "Cold Water Immersion": {"min_time": 3, "equipment": ["cold_water"], "difficulty": "hard", "benefits": ["inflammation", "recovery"]},
            "Gentle Stretching": {"min_time": 12, "equipment": ["mat"], "difficulty": "easy", "benefits": ["flexibility", "relaxation"]},
            "Massage Gun Therapy": {"min_time": 6, "equipment": ["massage_gun"], "difficulty": "medium", "benefits": ["circulation", "soreness"]},
            "Progressive Muscle Relaxation": {"min_time": 15, "equipment": [], "difficulty": "easy", "benefits": ["stress_relief", "relaxation"]},
            "Tennis Ball Foot Massage": {"min_time": 4, "equipment": ["tennis_ball"], "difficulty": "easy", "benefits": ["circulation", "foot_health"]},
            "Legs Up The Wall": {"min_time": 10, "equipment": ["wall"], "difficulty": "easy", "benefits": ["circulation", "relaxation"]},
            
            # Light Therapy Methods
            "Red Light Recovery Session": {"min_time": 15, "equipment": ["red_light_therapy"], "difficulty": "easy", "benefits": ["cellular_recovery", "inflammation", "muscle_fatigue"]},
            "Red Light Joint Therapy": {"min_time": 12, "equipment": ["red_light_therapy"], "difficulty": "easy", "benefits": ["joint_health", "inflammation", "range_of_motion"]},
            
            # Temperature Therapy Methods
            "Sauna Recovery Session": {"min_time": 20, "equipment": ["sauna"], "difficulty": "medium", "benefits": ["circulation", "relaxation", "cardiovascular_health"]},
            "Cold Tank Immersion": {"min_time": 4, "equipment": ["cold_tank"], "difficulty": "hard", "benefits": ["inflammation", "recovery", "soreness"]},
            "Hot Tank Therapy": {"min_time": 15, "equipment": ["hot_tank"], "difficulty": "easy", "benefits": ["circulation", "relaxation", "muscle_tension"]},
            "Contrast Hot-Cold Therapy": {"min_time": 12, "equipment": ["hot_tank", "cold_tank"], "difficulty": "hard", "benefits": ["circulation", "recovery", "waste_removal"]},
            "Targeted Heat Therapy": {"min_time": 15, "equipment": ["hot_pad"], "difficulty": "easy", "benefits": ["muscle_tension", "flexibility", "localized_relief"]},
            
            # Compression Therapy Methods
            "Normatec Leg Recovery": {"min_time": 30, "equipment": ["normatec"], "difficulty": "easy", "benefits": ["circulation", "lymphatic_drainage", "performance"]},
            "Normatec Arm Recovery": {"min_time": 20, "equipment": ["normatec"], "difficulty": "easy", "benefits": ["circulation", "upper_body_recovery", "fatigue"]},
            
            # Active Recovery Methods
            "Vibration Plate Recovery": {"min_time": 8, "equipment": ["vibration_plate"], "difficulty": "medium", "benefits": ["neuromuscular_control", "flexibility", "soreness"]},
            "Vibration Plate Stretching": {"min_time": 10, "equipment": ["vibration_plate"], "difficulty": "medium", "benefits": ["flexibility", "muscle_relaxation", "range_of_motion"]},
            
            # Percussion Therapy Methods
            "Hypervolt Percussion Therapy": {"min_time": 8, "equipment": ["hypervolt_gun"], "difficulty": "medium", "benefits": ["range_of_motion", "muscle_stiffness", "deep_tissue"]},
            
            # Stretching Methods - Shoulder
            "Shoulder Mobility Routine": {"min_time": 8, "equipment": [], "difficulty": "easy", "benefits": ["shoulder_health", "range_of_motion", "posture"]},
            "Band-Assisted Shoulder Stretches": {"min_time": 10, "equipment": ["resistance_bands"], "difficulty": "medium", "benefits": ["flexibility", "shoulder_mobility", "chest_opening"]},
            
            # Stretching Methods - Quadriceps
            "Quadriceps Stretch Routine": {"min_time": 6, "equipment": [], "difficulty": "easy", "benefits": ["hip_mobility", "lower_back_relief", "quad_flexibility"]},
            "Band-Assisted Quad Stretches": {"min_time": 8, "equipment": ["resistance_bands"], "difficulty": "medium", "benefits": ["hip_flexibility", "alignment", "balance"]},
            
            # Stretching Methods - Calf
            "Calf and Achilles Stretches": {"min_time": 6, "equipment": ["wall"], "difficulty": "easy", "benefits": ["ankle_mobility", "achilles_health", "injury_prevention"]},
            "Band-Assisted Calf Stretches": {"min_time": 7, "equipment": ["resistance_bands"], "difficulty": "easy", "benefits": ["calf_flexibility", "dorsiflexion", "progression"]},
            
            # Stretching Methods - Hamstring
            "Hamstring Flexibility Routine": {"min_time": 8, "equipment": [], "difficulty": "easy", "benefits": ["posterior_chain", "lower_back_health", "injury_prevention"]},
            "Band-Assisted Hamstring Stretches": {"min_time": 10, "equipment": ["resistance_bands"], "difficulty": "easy", "benefits": ["hamstring_length", "positioning", "gradual_progression"]},
            
            # Stretching Methods - Glute
            "Glute and Hip Stretch Routine": {"min_time": 8, "equipment": [], "difficulty": "medium", "benefits": ["hip_mobility", "lower_back_relief", "glute_activation"]},
            "Band-Assisted Glute Stretches": {"min_time": 10, "equipment": ["resistance_bands"], "difficulty": "medium", "benefits": ["piriformis_relief", "hip_positioning", "impingement_relief"]}
        }
    
    def predict(self, context, model_input):
        """Generate personalized recommendations"""
        predictions = []
        
        for index, row in model_input.iterrows():
            try:
                # Prepare features
                print(features)
                print('oogie boogie42')
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
                    "effectiveness_score": max(1.0, min(5.0, effectiveness_score)),  # Clamp values
                    "engagement_probability": max(0.0, min(1.0, engagement_prob)),
                    "recommended_methods": recommendations,
                    "optimal_duration": available_time,
                    "model_version": "v2.0"
                })
                
            except Exception as e:
                logger.error(f"Error generating prediction for row {index}: {e}")
                # Return default prediction
                predictions.append({
                    "user_index": index,
                    "effectiveness_score": 3.5,
                    "engagement_probability": 0.7,
                    "recommended_methods": [
                        {"method": "Deep Breathing", "confidence": 0.8, "reason": "Safe default option"}
                    ],
                    "optimal_duration": 15,
                    "model_version": "v2.0_fallback"
                })
        
        return predictions
    
    def _prepare_features(self, row):
        """Prepare features for prediction with proper error handling"""
        features = row.copy()
        
        # Encode categorical variables with fallback
        try:
            if 'fitness_level' in row and row['fitness_level'] in self.encoders['fitness_level'].classes_:
                features['fitness_level_encoded'] = self.encoders['fitness_level'].transform([row['fitness_level']])[0]
            else:
                features['fitness_level_encoded'] = self.encoders['fitness_level'].transform(['beginner'])[0]
        except:
            features['fitness_level_encoded'] = 0
        
        try:
            if 'age_group' in row and row['age_group'] in self.encoders['age_group'].classes_:
                features['age_group_encoded'] = self.encoders['age_group'].transform([row['age_group']])[0]
            else:
                features['age_group_encoded'] = self.encoders['age_group'].transform(['26-35'])[0]
        except:
            features['age_group_encoded'] = 1
        
        # Select and order features
        feature_cols = [
            'total_interactions', 'unique_sessions', 'avg_response_length',
            'total_sessions', 'avg_session_duration', 'avg_satisfaction',
            'avg_methods_completed', 'completion_efficiency',
            'fitness_level_encoded', 'age_group_encoded',
            'preferred_locations_count', 'equipment_variety'
        ]
        
        # Ensure all features exist with defaults
        for col in feature_cols:
            if col not in features:
                features[col] = self._get_default_value(col)
        
        # Scale features
        feature_array = features[feature_cols].values.reshape(1, -1)
        feature_array = self.scaler.transform(feature_array)
        
        return feature_array
    
    def _get_default_value(self, feature_name: str) -> float:
        """Get default value for missing features"""
        defaults = {
            'total_interactions': 0,
            'unique_sessions': 0,
            'avg_response_length': 100,
            'total_sessions': 0,
            'avg_session_duration': 15,
            'avg_satisfaction': 3.0,
            'avg_methods_completed': 2,
            'completion_efficiency': 0.6,
            'fitness_level_encoded': 0,
            'age_group_encoded': 1,
            'preferred_locations_count': 1,
            'equipment_variety': 1
        }
        return defaults.get(feature_name, 0)
    
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
                "reason": reason,
                "estimated_duration": self.recovery_methods[method]["min_time"],
                "required_equipment": self.recovery_methods[method]["equipment"]
            })
        
        return recommendations
    
    def _calculate_method_score(self, method, info, effectiveness_score, 
                               engagement_prob, available_time, available_equipment, fitness_level):
        """Calculate score for a specific method with improved logic"""
        score = 50  # Base score
        
        # Effectiveness component (0-30 points)
        if effectiveness_score >= 4.0:
            if info['difficulty'] in ['medium', 'hard']:
                score += 30  # High effectiveness users can handle challenging methods
            else:
                score += 25
        elif effectiveness_score >= 3.5:
            if info['difficulty'] in ['easy', 'medium']:
                score += 25
            else:
                score += 15  # Moderate users get less benefit from hard methods
        else:
            if info['difficulty'] == 'easy':
                score += 25  # Low effectiveness users need easy methods
            else:
                score += 10
        
        # Engagement component (0-20 points)
        score += engagement_prob * 20
        
        # Time availability (0-20 points)
        if available_time >= info['min_time']:
            score += 20
        else:
            # Penalize methods that don't fit time constraints
            score += max(0, 20 - (info['min_time'] - available_time) * 3)
        
        # Equipment availability (0-15 points)
        if not info['equipment']:  # No equipment needed
            score += 15
        elif all(eq in available_equipment for eq in info['equipment']):
            score += 15
        elif any(eq in available_equipment for eq in info['equipment']):
            score += 7
        else:
            score -= 10  # Penalize if equipment not available
        
        # Fitness level match (0-15 points)
        if fitness_level == 'beginner' and info['difficulty'] == 'easy':
            score += 15
        elif fitness_level == 'intermediate' and info['difficulty'] in ['easy', 'medium']:
            score += 15
        elif fitness_level == 'advanced':
            score += 15  # Advanced users can handle any difficulty
        else:
            score += 5
        
        return max(0, score)  # Ensure non-negative score
    
    def _get_recommendation_reason(self, method, effectiveness_score, engagement_prob, fitness_level):
        """Generate contextual reason for recommendation"""
        reasons = []
        
        if effectiveness_score >= 4.0:
            reasons.append("optimized for high-performing users")
        elif effectiveness_score >= 3.5:
            reasons.append("well-suited to your experience level")
        else:
            reasons.append("gentle introduction to recovery")
        
        if engagement_prob >= 0.8:
            reasons.append("matches your consistency pattern")
        
        if fitness_level == 'beginner':
            reasons.append("beginner-friendly")
        elif fitness_level == 'advanced':
            reasons.append("appropriate for advanced users")
        
        # Method-specific reasons
        method_reasons = {
            "Deep Breathing": "requires no equipment and provides universal benefits",
            "Foam Rolling - Legs": "highly effective for muscle recovery",
            "Cold Water Immersion": "advanced technique with proven benefits",
            "Gentle Stretching": "safe and accessible for all levels",
            "Legs Up The Wall": "passive recovery with immediate benefits"
        }
        
        if method in method_reasons:
            reasons.append(method_reasons[method])
        
        return ", ".join(reasons[:3])  # Limit to 3 reasons for brevity

def load_or_create_training_data(config: ModelConfig):
    """Load training data from features or create synthetic data"""
    features_path = LOCAL_DATA_PATH / "user_features.csv"
    
    if features_path.exists():
        logger.info("Loading features from CSV...")
        df = pd.read_csv(features_path)
        logger.info(f"Loaded {len(df)} user features")
        return df
    else:
        logger.info("Creating synthetic training data...")
        generator = SyntheticDataGenerator(config)
        return generator.create_synthetic_data()

def train_models(config: ModelConfig):
    """Train all models with MLflow tracking"""
    mlflow.set_experiment("local_recovery_recommendations_v2")
    
    # Load or create data
    df = load_or_create_training_data(config)
    
    # Generate synthetic data with proper targets
    if 'effectiveness' not in df.columns:
        generator = SyntheticDataGenerator(config)
        y_effectiveness, y_engagement = generator.create_target_variables(df)
        df['effectiveness'] = y_effectiveness
        df['high_engagement'] = y_engagement
    
    # Save training data
    df.to_csv(LOCAL_DATA_PATH / "training_data_v2.csv", index=False)
    logger.info(f"Training data saved to {LOCAL_DATA_PATH / 'training_data_v2.csv'}")
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Prepare features
    logger.info("Preparing features...")
    X = trainer.prepare_features(df, fit_transformers=True)
    
    # Train effectiveness model
    logger.info("Training effectiveness predictor...")
    with mlflow.start_run(run_name="effectiveness_predictor_v2"):
        y_effectiveness = df['effectiveness']
        
        # Log data information
        mlflow.log_params({
            "n_samples": len(df),
            "n_features": len(X.columns),
            "target_mean": y_effectiveness.mean(),
            "target_std": y_effectiveness.std()
        })
        
        effectiveness_model = trainer.train_effectiveness_model(X, y_effectiveness)
        
        # Log and save model
        mlflow.sklearn.log_model(effectiveness_model, "model")
        with open(LOCAL_MODELS_PATH / "effectiveness_model.pkl", 'wb') as f:
            pickle.dump(effectiveness_model, f)
    
    # Train engagement model
    logger.info("Training engagement predictor...")
    with mlflow.start_run(run_name="engagement_predictor_v2"):
        y_engagement = df['high_engagement']
        
        # Log data information
        mlflow.log_params({
            "n_samples": len(df),
            "n_features": len(X.columns),
            "positive_class_ratio": y_engagement.mean()
        })
        
        engagement_model = trainer.train_engagement_model(X, y_engagement)
        
        # Log and save model
        mlflow.sklearn.log_model(engagement_model, "model")
        with open(LOCAL_MODELS_PATH / "engagement_model.pkl", 'wb') as f:
            pickle.dump(engagement_model, f)
    
    # Create and save combined model
    logger.info("Creating combined recommendation model...")
    with mlflow.start_run(run_name="combined_recommendation_model_v2"):
        # Create combined model
        combined_model = RecoveryRecommendationModel(
            effectiveness_model=effectiveness_model,
            engagement_model=engagement_model,
            encoders=trainer.encoders,
            scaler=trainer.scaler
        )
        
        # Save artifacts
        artifacts = {
            'encoders': trainer.encoders,
            'scaler': trainer.scaler,
            'feature_columns': X.columns.tolist()
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
            'fitness_level': 'intermediate',
            'age_group': '26-35',
            'preferred_locations_count': 2,
            'equipment_variety': 3,
            'available_time': 20,
            'available_equipment': ['foam_roller', 'mat']
        }])
        
        predictions = combined_model.predict(None, test_input)
        
        mlflow.log_text(str(predictions[0]), "sample_prediction.txt")
        
        # Log model performance summary
        mlflow.log_params({
            "model_version": "v2.0",
            "features_used": len(X.columns),
            "training_samples": len(df)
        })
        
        logger.info("Combined model created and logged")
        logger.info(f"Sample prediction: {predictions[0]}")
    
    return effectiveness_model, engagement_model, artifacts

if __name__ == "__main__":
    logger.info("Starting improved model training pipeline...")
    
    # Load configuration
    config = ModelConfig()
    
    # Train models
    effectiveness_model, engagement_model, artifacts = train_models(config)
    
    logger.info("\nModel training completed!")
    logger.info(f"Models saved to: {LOCAL_MODELS_PATH}")
    logger.info(f"MLflow runs saved to: {LOCAL_MLRUNS_PATH}")
    logger.info("\nTo view MLflow UI, run: mlflow ui --backend-store-uri file://./mlruns")

