"""
Local Feature Engineering Script - Fixed Version
Resolves DuckDB column mismatch issue
Enhanced with comprehensive synthetic data generation
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import sqlite3
import duckdb
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
LOCAL_DB_PATH = Path("./databases")
LOCAL_DATA_PATH = Path("./data")

class LocalFeatureEngineering:
    """Local feature engineering pipeline with synthetic data generation"""
    
    def __init__(self):
        self.sqlite_path = LOCAL_DB_PATH / "recoveredge.db"
        self.duckdb_path = LOCAL_DB_PATH / "analytics.duckdb"
        self.recovery_methods = [
            "Deep Breathing", "Foam Rolling", "Cold Therapy", "Stretching",
            "Massage Gun Therapy", "Progressive Muscle Relaxation", 
            "Tennis Ball Foot Massage", "Legs Up The Wall", "Ice Bath",
            "Meditation", "Gentle Yoga", "Heat Therapy"
        ]
        self.locations = ["home", "gym", "office", "outdoor", "studio"]
        self.equipment_options = [
            "foam_roller", "mat", "massage_gun", "tennis_ball", "ice_bath",
            "heating_pad", "resistance_bands", "yoga_blocks", "meditation_app"
        ]
        
    def initialize_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # Create user_interactions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_interactions (
            interaction_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_id TEXT,
            interaction_type TEXT,
            timestamp DATETIME,
            content TEXT,
            metadata TEXT,
            location TEXT,
            equipment_used TEXT
        )
        """)
        
        # Create recovery_sessions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS recovery_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            planned_duration INTEGER,
            actual_duration INTEGER,
            completed_methods TEXT,
            skipped_methods TEXT,
            location TEXT,
            equipment_used TEXT,
            satisfaction_rating INTEGER,
            created_at DATETIME,
            completed_at DATETIME
        )
        """)
        
        # Create method_effectiveness table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS method_effectiveness (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            method_name TEXT NOT NULL,
            user_id TEXT NOT NULL,
            effectiveness_rating REAL,
            duration_completed INTEGER,
            reported_benefits TEXT,
            side_effects TEXT,
            timestamp DATETIME,
            user_fitness_level TEXT,
            user_age_group TEXT
        )
        """)
        
        conn.commit()
        conn.close()
        
        # Create user_features table in DuckDB with exact column order
        duckdb_conn = duckdb.connect(str(self.duckdb_path))
        
        # Drop table if exists to ensure clean schema
        duckdb_conn.execute("DROP TABLE IF EXISTS user_features")
        
        # Create table with exact columns we'll provide
        duckdb_conn.execute("""
        CREATE TABLE user_features (
            user_id TEXT PRIMARY KEY,
            total_interactions INTEGER DEFAULT 0,
            unique_sessions INTEGER DEFAULT 0,
            avg_response_length REAL DEFAULT 0.0,
            preferred_locations TEXT DEFAULT '[]',
            equipment_used TEXT DEFAULT '[]',
            preferred_locations_count INTEGER DEFAULT 0,
            equipment_variety INTEGER DEFAULT 0,
            total_sessions INTEGER DEFAULT 0,
            avg_session_duration REAL DEFAULT 0.0,
            avg_satisfaction REAL DEFAULT 0.0,
            avg_methods_completed REAL DEFAULT 0.0,
            completion_efficiency REAL DEFAULT 0.0,
            avg_method_effectiveness REAL DEFAULT 0.0,
            tried_methods TEXT DEFAULT '[]',
            fitness_level TEXT DEFAULT 'beginner',
            age_group TEXT DEFAULT 'unknown',
            feature_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        duckdb_conn.close()
        logger.info("Database initialized with all required tables")
    
    def check_data_exists(self, min_users=50):
        """Check if sufficient data exists in database"""
        if not self.sqlite_path.exists():
            return False
            
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # Check user count across tables
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_interactions")
        result = cursor.fetchone()
        interaction_users = result[0] if result else 0
        
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM recovery_sessions")
        result = cursor.fetchone()
        session_users = result[0] if result else 0
        
        conn.close()
        
        return max(interaction_users, session_users) >= min_users
    
    def generate_synthetic_data(self, n_users=200, days_back=90):
        """Generate comprehensive synthetic raw data"""
        logger.info(f"Generating synthetic data for {n_users} users over {days_back} days...")
        
        # Clear existing data
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_interactions")
        cursor.execute("DELETE FROM recovery_sessions")
        cursor.execute("DELETE FROM method_effectiveness")
        conn.commit()
        
        # Generate user profiles
        np.random.seed(42)
        user_profiles = self._generate_user_profiles(n_users)
        
        # Generate data for each user
        for user_id, profile in user_profiles.items():
            self._generate_user_interactions(cursor, user_id, profile, days_back)
            self._generate_recovery_sessions(cursor, user_id, profile, days_back)
            self._generate_method_effectiveness(cursor, user_id, profile, days_back)
        
        conn.commit()
        conn.close()
        logger.info(f"Synthetic data generation completed for {n_users} users")
    
    def _generate_user_profiles(self, n_users):
        """Generate realistic user profiles"""
        profiles = {}
        fitness_levels = ['beginner', 'intermediate', 'advanced']
        age_groups = ['18-25', '26-35', '36-45', '46+']
        
        for i in range(n_users):
            user_id = f"user_{i:03d}"
            
            # User characteristics
            fitness_level = np.random.choice(fitness_levels, p=[0.5, 0.35, 0.15])
            age_group = np.random.choice(age_groups, p=[0.25, 0.35, 0.25, 0.15])
            
            # Behavior patterns based on fitness level
            if fitness_level == 'beginner':
                activity_level = np.random.uniform(0.3, 0.7)
                completion_tendency = np.random.uniform(0.4, 0.8)
            elif fitness_level == 'intermediate':
                activity_level = np.random.uniform(0.6, 0.9)
                completion_tendency = np.random.uniform(0.7, 0.95)
            else:  # advanced
                activity_level = np.random.uniform(0.8, 1.0)
                completion_tendency = np.random.uniform(0.8, 1.0)
            
            profiles[user_id] = {
                'fitness_level': fitness_level,
                'age_group': age_group,
                'activity_level': activity_level,
                'completion_tendency': completion_tendency,
                'preferred_locations': np.random.choice(self.locations, 
                                                       size=np.random.randint(1, 4), 
                                                       replace=False).tolist(),
                'available_equipment': np.random.choice(self.equipment_options,
                                                       size=np.random.randint(2, 6),
                                                       replace=False).tolist(),
                'preferred_methods': np.random.choice(self.recovery_methods,
                                                     size=np.random.randint(3, 7),
                                                     replace=False).tolist()
            }
        
        return profiles
    
    def _generate_user_interactions(self, cursor, user_id, profile, days_back):
        """Generate realistic user interactions"""
        # Number of interactions based on activity level
        n_interactions = int(np.random.poisson(profile['activity_level'] * 15))
        
        session_counter = 0
        for i in range(n_interactions):
            # Group interactions into sessions
            if i % np.random.randint(2, 5) == 0:
                session_counter += 1
            
            session_id = f"{user_id}_session_{session_counter:03d}"
            
            # Generate interaction content
            interaction_types = ['chat', 'query', 'feedback', 'request']
            interaction_type = np.random.choice(interaction_types, p=[0.4, 0.3, 0.2, 0.1])
            
            # Generate realistic content based on interaction type
            if interaction_type == 'chat':
                content = f"How do I improve my {np.random.choice(['flexibility', 'recovery', 'soreness', 'performance'])}?"
            elif interaction_type == 'query':
                method = np.random.choice(profile['preferred_methods'])
                content = f"Tell me more about {method} for recovery"
            elif interaction_type == 'feedback':
                content = f"The {np.random.choice(profile['preferred_methods'])} session was helpful"
            else:  # request
                content = f"Can you recommend something for {np.random.choice(['post-workout', 'morning', 'evening'])} recovery?"
            
            # Metadata with response length
            response_length = int(np.random.normal(100, 30))
            if response_length < 20:
                response_length = 20
            
            metadata = json.dumps({
                'response_length': response_length,
                'satisfaction': np.random.uniform(3, 5),
                'helpful': str(np.random.choice([True, False], p=[0.8, 0.2]))
            })
            
            # Location and equipment
            location = np.random.choice(profile['preferred_locations'])
            equipment_used = json.dumps(
                np.random.choice(profile['available_equipment'], 
                               size=np.random.randint(0, 3), 
                               replace=False).tolist()
            )
            
            # Timestamp
            timestamp = datetime.now() - timedelta(
                days=np.random.randint(0, days_back),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            
            cursor.execute("""
            INSERT INTO user_interactions 
            (interaction_id, user_id, session_id, interaction_type, timestamp, 
             content, metadata, location, equipment_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"{user_id}_interaction_{i:03d}",
                user_id,
                session_id,
                interaction_type,
                timestamp,
                content,
                metadata,
                location,
                equipment_used
            ))
    
    def _generate_recovery_sessions(self, cursor, user_id, profile, days_back):
        """Generate realistic recovery sessions"""
        # Number of sessions based on activity level
        n_sessions = int(np.random.poisson(profile['activity_level'] * 10))
        
        for i in range(n_sessions):
            session_id = f"{user_id}_session_{i:03d}"
            
            # Session duration
            planned_duration = np.random.randint(10, 45)
            
            # Actual duration based on completion tendency
            if np.random.random() < profile['completion_tendency']:
                actual_duration = int(planned_duration * np.random.uniform(0.8, 1.2))
            else:
                actual_duration = int(planned_duration * np.random.uniform(0.3, 0.8))
            
            # Methods selection
            n_methods = np.random.randint(1, 4)
            selected_methods = np.random.choice(profile['preferred_methods'], 
                                              size=n_methods, 
                                              replace=False).tolist()
            
            # Completion based on tendency
            if np.random.random() < profile['completion_tendency']:
                completed_methods = selected_methods
                skipped_methods = []
            else:
                n_completed = np.random.randint(0, len(selected_methods))
                completed_methods = selected_methods[:n_completed]
                skipped_methods = selected_methods[n_completed:]

            sample_size = min(np.random.randint(1, 4), len(profile['available_equipment']))
            
            # Location and equipment
            location = np.random.choice(profile['preferred_locations'])
            equipment_used = json.dumps(
                np.random.choice(profile['available_equipment'],
                               size=sample_size,
                               replace=False).tolist()
            )
            
            # Satisfaction rating
            if len(completed_methods) == len(selected_methods):
                satisfaction_rating = np.random.randint(4, 6)
            else:
                satisfaction_rating = np.random.randint(2, 4)
            
            # Timestamps
            created_at = datetime.now() - timedelta(
                days=np.random.randint(0, days_back),
                hours=np.random.randint(6, 22)
            )
            completed_at = created_at + timedelta(minutes=actual_duration)
            
            cursor.execute("""
            INSERT INTO recovery_sessions
            (session_id, user_id, planned_duration, actual_duration,
             completed_methods, skipped_methods, location, equipment_used,
             satisfaction_rating, created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                user_id,
                planned_duration,
                actual_duration,
                json.dumps(completed_methods),
                json.dumps(skipped_methods),
                location,
                equipment_used,
                satisfaction_rating,
                created_at,
                completed_at
            ))
    
    def _generate_method_effectiveness(self, cursor, user_id, profile, days_back):
        """Generate method effectiveness ratings"""
        # Generate ratings for methods the user has tried
        for method in profile['preferred_methods']:
            # Number of times they've rated this method
            n_ratings = np.random.randint(1, 5)
            
            for rating_idx in range(n_ratings):
                # Effectiveness rating influenced by fitness level
                if profile['fitness_level'] == 'advanced':
                    effectiveness_rating = np.random.uniform(3.5, 5.0)
                elif profile['fitness_level'] == 'intermediate':
                    effectiveness_rating = np.random.uniform(3.0, 4.5)
                else:  # beginner
                    effectiveness_rating = np.random.uniform(2.5, 4.0)
                
                # Duration completed
                duration_completed = np.random.randint(5, 30)
                
                # Benefits and side effects
                all_benefits = [
                    "reduced_soreness", "improved_flexibility", "better_sleep",
                    "reduced_stress", "improved_mood", "faster_recovery",
                    "increased_energy", "pain_relief"
                ]
                
                reported_benefits = json.dumps(
                    np.random.choice(all_benefits, 
                                   size=np.random.randint(1, 4),
                                   replace=False).tolist()
                )
                
                # Side effects (rare)
                side_effects = json.dumps([]) if np.random.random() > 0.1 else json.dumps(["mild_discomfort"])
                
                # Timestamp
                timestamp = datetime.now() - timedelta(
                    days=np.random.randint(0, days_back),
                    hours=np.random.randint(0, 24)
                )
                
                cursor.execute("""
                INSERT INTO method_effectiveness
                (method_name, user_id, effectiveness_rating, duration_completed,
                 reported_benefits, side_effects, timestamp, 
                 user_fitness_level, user_age_group)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    method,
                    user_id,
                    effectiveness_rating,
                    duration_completed,
                    reported_benefits,
                    side_effects,
                    timestamp,
                    profile['fitness_level'],
                    profile['age_group']
                ))
    
    def run_pipeline(self, force_regenerate=False):
        """Run the complete feature engineering pipeline"""
        logger.info("Starting feature engineering pipeline...")
        
        # Initialize database
        self.initialize_database()
        
        # Check if we need to generate synthetic data
        if force_regenerate or not self.check_data_exists():
            logger.info("Generating synthetic data...")
            self.generate_synthetic_data()
        else:
            logger.info("Using existing data from database")
        
        # Step 1: Load data from SQLite
        interactions_df = self.load_user_interactions()
        sessions_df = self.load_recovery_sessions()
        effectiveness_df = self.load_method_effectiveness()
        
        # Step 2: Create features
        user_features = self.create_user_features(interactions_df)
        session_features = self.create_session_features(sessions_df)
        effectiveness_features = self.create_effectiveness_features(effectiveness_df)
        
        # Step 3: Combine features
        final_features = self.combine_features(
            user_features, session_features, effectiveness_features
        )
        
        # Step 4: Save to DuckDB
        self.save_features(final_features)
        
        logger.info(f"Feature engineering completed for {len(final_features)} users")
        
        return final_features
    
    def load_user_interactions(self) -> pd.DataFrame:
        """Load user interactions from SQLite"""
        conn = sqlite3.connect(self.sqlite_path)
        
        query = """
        SELECT 
            user_id,
            session_id,
            interaction_type,
            timestamp,
            content,
            metadata,
            location,
            equipment_used
        FROM user_interactions
        WHERE timestamp > datetime('now', '-90 days')
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            return df
        
        # Parse JSON fields safely
        df['metadata'] = df['metadata'].apply(
            lambda x: json.loads(x) if x else {}
        )
        df['equipment_used'] = df['equipment_used'].apply(
            lambda x: json.loads(x) if x else []
        )
        
        return df
    
    def load_recovery_sessions(self) -> pd.DataFrame:
        """Load recovery sessions from SQLite"""
        conn = sqlite3.connect(self.sqlite_path)
        
        query = """
        SELECT 
            session_id,
            user_id,
            planned_duration,
            actual_duration,
            completed_methods,
            skipped_methods,
            location,
            equipment_used,
            satisfaction_rating,
            created_at,
            completed_at
        FROM recovery_sessions
        WHERE created_at > datetime('now', '-90 days')
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            return df
        
        # Parse JSON fields safely
        df['completed_methods'] = df['completed_methods'].apply(
            lambda x: json.loads(x) if x else []
        )
        df['skipped_methods'] = df['skipped_methods'].apply(
            lambda x: json.loads(x) if x else []
        )
        df['equipment_used'] = df['equipment_used'].apply(
            lambda x: json.loads(x) if x else []
        )
        
        return df
    
    def load_method_effectiveness(self) -> pd.DataFrame:
        """Load method effectiveness data from SQLite"""
        conn = sqlite3.connect(self.sqlite_path)
        
        query = """
        SELECT 
            method_name,
            user_id,
            effectiveness_rating,
            duration_completed,
            reported_benefits,
            side_effects,
            timestamp,
            user_fitness_level,
            user_age_group
        FROM method_effectiveness
        WHERE timestamp > datetime('now', '-90 days')
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            return df
        
        # Parse JSON fields safely
        df['reported_benefits'] = df['reported_benefits'].apply(
            lambda x: json.loads(x) if x else []
        )
        df['side_effects'] = df['side_effects'].apply(
            lambda x: json.loads(x) if x else []
        )
        
        return df
    
    def create_user_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create user behavior features"""
        if interactions_df.empty:
            return pd.DataFrame(columns=[
                'user_id', 'total_interactions', 'unique_sessions',
                'avg_response_length', 'preferred_locations', 'equipment_used',
                'preferred_locations_count', 'equipment_variety'
            ])
        
        # Extract response length from metadata
        interactions_df['response_length'] = interactions_df['metadata'].apply(
            lambda x: x.get('response_length', 0) if isinstance(x, dict) else 0
        )
        
        # Aggregate features
        user_features = interactions_df.groupby('user_id').agg({
            'session_id': ['count', 'nunique'],
            'response_length': 'mean',
            'location': lambda x: list(set(x)),
            'equipment_used': lambda x: list(set([item for sublist in x for item in sublist if item]))
        }).reset_index()
        
        # Flatten column names
        user_features.columns = [
            'user_id', 
            'total_interactions', 
            'unique_sessions',
            'avg_response_length',
            'preferred_locations',
            'equipment_used'
        ]
        
        # Add derived features
        user_features['preferred_locations_count'] = user_features['preferred_locations'].apply(len)
        user_features['equipment_variety'] = user_features['equipment_used'].apply(len)
        
        return user_features
    
    def create_session_features(self, sessions_df: pd.DataFrame) -> pd.DataFrame:
        """Create session-based features"""
        if sessions_df.empty:
            return pd.DataFrame(columns=[
                'user_id', 'total_sessions', 'avg_session_duration',
                'avg_satisfaction', 'avg_methods_completed', 'completion_efficiency'
            ])
        
        # Calculate completion metrics
        sessions_df['methods_completed_count'] = sessions_df['completed_methods'].apply(len)
        sessions_df['methods_skipped_count'] = sessions_df['skipped_methods'].apply(len)
        sessions_df['completion_rate'] = sessions_df.apply(
            lambda x: x['methods_completed_count'] / (x['methods_completed_count'] + x['methods_skipped_count']) 
            if (x['methods_completed_count'] + x['methods_skipped_count']) > 0 else 0, 
            axis=1
        )
        
        # Aggregate features
        session_features = sessions_df.groupby('user_id').agg({
            'session_id': 'count',
            'actual_duration': 'mean',
            'satisfaction_rating': 'mean',
            'methods_completed_count': 'mean',
            'completion_rate': 'mean'
        }).reset_index()
        
        session_features.columns = [
            'user_id',
            'total_sessions',
            'avg_session_duration',
            'avg_satisfaction',
            'avg_methods_completed',
            'completion_efficiency'
        ]
        
        return session_features
    
    def create_effectiveness_features(self, effectiveness_df: pd.DataFrame) -> pd.DataFrame:
        """Create method effectiveness features"""
        if effectiveness_df.empty:
            return pd.DataFrame(columns=[
                'user_id', 'avg_method_effectiveness', 'tried_methods',
                'fitness_level', 'age_group'
            ])
        
        # Aggregate features
        effectiveness_features = effectiveness_df.groupby('user_id').agg({
            'effectiveness_rating': 'mean',
            'method_name': lambda x: list(x),
            'user_fitness_level': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'beginner',
            'user_age_group': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
        }).reset_index()
        
        effectiveness_features.columns = [
            'user_id',
            'avg_method_effectiveness',
            'tried_methods',
            'fitness_level',
            'age_group'
        ]
        
        return effectiveness_features
    
    def combine_features(self, user_features: pd.DataFrame, 
                        session_features: pd.DataFrame,
                        effectiveness_features: pd.DataFrame) -> pd.DataFrame:
        """Combine all feature sets with exact column order"""
        
        # Get all unique user IDs
        all_user_ids = set()
        if not user_features.empty:
            all_user_ids.update(user_features['user_id'])
        if not session_features.empty:
            all_user_ids.update(session_features['user_id'])
        if not effectiveness_features.empty:
            all_user_ids.update(effectiveness_features['user_id'])
        
        if not all_user_ids:
            return pd.DataFrame()
        
        # Create base DataFrame with all users
        final_features = pd.DataFrame({'user_id': list(all_user_ids)})
        
        # Merge features in order
        if not user_features.empty:
            final_features = final_features.merge(user_features, on='user_id', how='left')
        else:
            # Add empty columns for user features
            for col in ['total_interactions', 'unique_sessions', 'avg_response_length', 
                       'preferred_locations', 'equipment_used', 'preferred_locations_count', 
                       'equipment_variety']:
                if col not in final_features.columns:
                    if col in ['preferred_locations', 'equipment_used']:
                        final_features[col] = [[] for _ in range(len(final_features))]
                    else:
                        final_features[col] = 0
            
        if not session_features.empty:
            final_features = final_features.merge(session_features, on='user_id', how='left')
        else:
            # Add empty columns for session features
            for col in ['total_sessions', 'avg_session_duration', 'avg_satisfaction', 
                       'avg_methods_completed', 'completion_efficiency']:
                if col not in final_features.columns:
                    final_features[col] = 0.0
            
        if not effectiveness_features.empty:
            final_features = final_features.merge(effectiveness_features, on='user_id', how='left')
        else:
            # Add empty columns for effectiveness features
            if 'avg_method_effectiveness' not in final_features.columns:
                final_features['avg_method_effectiveness'] = 0.0
            if 'tried_methods' not in final_features.columns:
                final_features['tried_methods'] = [[] for _ in range(len(final_features))]
            if 'fitness_level' not in final_features.columns:
                final_features['fitness_level'] = 'beginner'
            if 'age_group' not in final_features.columns:
                final_features['age_group'] = 'unknown'
        
        # Fill missing values
        numeric_columns = final_features.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            final_features[col] = final_features[col].fillna(0)
        
        # Fill categorical columns
        final_features['fitness_level'] = final_features['fitness_level'].fillna('beginner')
        final_features['age_group'] = final_features['age_group'].fillna('unknown')
        
        # Fill list columns
        for col in ['preferred_locations', 'equipment_used', 'tried_methods']:
            if col in final_features.columns:
                final_features[col] = final_features[col].apply(
                    lambda x: x if isinstance(x, list) else []
                )
        
        # Add timestamp
        final_features['feature_timestamp'] = datetime.now()
        
        # Ensure exact column order to match database schema
        expected_columns = [
            'user_id', 'total_interactions', 'unique_sessions', 'avg_response_length',
            'preferred_locations', 'equipment_used', 'preferred_locations_count',
            'equipment_variety', 'total_sessions', 'avg_session_duration',
            'avg_satisfaction', 'avg_methods_completed', 'completion_efficiency',
            'avg_method_effectiveness', 'tried_methods', 'fitness_level',
            'age_group', 'feature_timestamp'
        ]
        
        # Reorder columns to match schema
        final_features = final_features[expected_columns]
        
        return final_features
    
    def save_features(self, features_df: pd.DataFrame):
        """Save features to DuckDB with proper column mapping"""
        if features_df.empty:
            logger.warning("No features to save")
            return
        
        conn = duckdb.connect(str(self.duckdb_path))
        
        # Clear existing features
        conn.execute("DELETE FROM user_features")
        
        # Prepare data for insertion
        features_df = features_df.copy()
        
        # Convert lists to JSON strings for storage
        for col in ['preferred_locations', 'equipment_used', 'tried_methods']:
            if col in features_df.columns:
                features_df[col] = features_df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, list) else json.dumps([])
                )
        
        # Verify column count matches
        logger.info(f"DataFrame columns: {len(features_df.columns)}")
        logger.info(f"DataFrame shape: {features_df.shape}")
        
        # Get table schema to verify
        try:
            schema_info = conn.execute("DESCRIBE user_features").fetchall()
            logger.info(f"Table has {len(schema_info)} columns")
            logger.info("Table schema:")
            for col_info in schema_info:
                logger.info(f"  {col_info}")
        except Exception as e:
            logger.warning(f"Could not get table schema: {e}")
        
        # Insert features using explicit column specification
        try:
            # Convert DataFrame to the format DuckDB expects
            conn.execute("INSERT INTO user_features SELECT * FROM features_df")
            logger.info(f"Successfully saved {len(features_df)} user features to DuckDB")
        except Exception as e:
            logger.error(f"Error inserting features: {e}")
            
            # Try alternative insertion method with explicit columns
            try:
                logger.info("Attempting alternative insertion method...")
                
                # Create a temporary view and insert from it
                conn.register('temp_features', features_df)
                
                insert_query = """
                INSERT INTO user_features (
                    user_id, total_interactions, unique_sessions, avg_response_length,
                    preferred_locations, equipment_used, preferred_locations_count,
                    equipment_variety, total_sessions, avg_session_duration,
                    avg_satisfaction, avg_methods_completed, completion_efficiency,
                    avg_method_effectiveness, tried_methods, fitness_level,
                    age_group, feature_timestamp
                )
                SELECT 
                    user_id, total_interactions, unique_sessions, avg_response_length,
                    preferred_locations, equipment_used, preferred_locations_count,
                    equipment_variety, total_sessions, avg_session_duration,
                    avg_satisfaction, avg_methods_completed, completion_efficiency,
                    avg_method_effectiveness, tried_methods, fitness_level,
                    age_group, feature_timestamp
                FROM temp_features
                """
                
                conn.execute(insert_query)
                logger.info(f"Successfully saved {len(features_df)} user features using alternative method")
                
            except Exception as e2:
                logger.error(f"Alternative insertion also failed: {e2}")
                
                # Last resort: insert row by row
                logger.info("Attempting row-by-row insertion...")
                try:
                    for idx, row in features_df.iterrows():
                        conn.execute("""
                        INSERT INTO user_features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, tuple(row))
                    
                    logger.info(f"Successfully saved {len(features_df)} user features row by row")
                    
                except Exception as e3:
                    logger.error(f"Row-by-row insertion failed: {e3}")
                    logger.error("Saving features to CSV as backup...")
                    features_df.to_csv(LOCAL_DATA_PATH / "failed_features_backup.csv", index=False)
        
        finally:
            conn.close()

if __name__ == "__main__":
    # Ensure directories exist
    LOCAL_DB_PATH.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    # Run feature engineering with synthetic data generation
    feature_eng = LocalFeatureEngineering()
    features = feature_eng.run_pipeline(force_regenerate=True)  # Set to True to regenerate data
    
    if not features.empty:
        print("\nFeature Summary:")
        print(f"Total users: {len(features)}")
        print(f"\nFeature columns ({len(features.columns)}): {list(features.columns)}")
        print(f"\nSample features:")
        print(features.head())
        
        # Save to CSV for inspection
        features.to_csv(LOCAL_DATA_PATH / "user_features.csv", index=False)
        print(f"\nFeatures saved to {LOCAL_DATA_PATH / 'user_features.csv'}")
        
        # Print some statistics
        print(f"\nFeature Statistics:")
        print(f"Average interactions per user: {features['total_interactions'].mean():.1f}")
        print(f"Average sessions per user: {features['total_sessions'].mean():.1f}")
        print(f"Average satisfaction: {features['avg_satisfaction'].mean():.2f}")
        print(f"Fitness level distribution:")
        print(features['fitness_level'].value_counts())
        
    else:
        print("No features generated. Check database initialization.")