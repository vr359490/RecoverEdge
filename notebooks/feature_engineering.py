"""
Local Feature Engineering Script
Replaces Databricks notebook with local processing
"""

import pandas as pd
import numpy as np
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
    """Local feature engineering pipeline"""
    
    def __init__(self):
        self.sqlite_path = LOCAL_DB_PATH / "recoveredge.db"
        self.duckdb_path = LOCAL_DB_PATH / "analytics.duckdb"
        
    def run_pipeline(self):
        """Run the complete feature engineering pipeline"""
        logger.info("Starting feature engineering pipeline...")
        
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
        
        # Parse JSON fields
        df['metadata'] = df['metadata'].apply(lambda x: eval(x) if x else {})
        df['equipment_used'] = df['equipment_used'].apply(lambda x: eval(x) if x else [])
        
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
        
        # Parse JSON fields
        df['completed_methods'] = df['completed_methods'].apply(lambda x: eval(x) if x else [])
        df['skipped_methods'] = df['skipped_methods'].apply(lambda x: eval(x) if x else [])
        df['equipment_used'] = df['equipment_used'].apply(lambda x: eval(x) if x else [])
        
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
        
        # Parse JSON fields
        df['reported_benefits'] = df['reported_benefits'].apply(lambda x: eval(x) if x else [])
        df['side_effects'] = df['side_effects'].apply(lambda x: eval(x) if x else [])
        
        return df
    
    def create_user_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create user behavior features"""
        if interactions_df.empty:
            return pd.DataFrame()
        
        # Extract response length from metadata
        interactions_df['response_length'] = interactions_df['metadata'].apply(
            lambda x: x.get('response_length', 0) if isinstance(x, dict) else 0
        )
        
        # Aggregate features
        user_features = interactions_df.groupby('user_id').agg({
            'session_id': ['count', 'nunique'],
            'response_length': 'mean',
            'location': lambda x: list(set(x)),
            'equipment_used': lambda x: list(set([item for sublist in x for item in sublist]))
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
            return pd.DataFrame()
        
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
            return pd.DataFrame()
        
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
        """Combine all feature sets"""
        # Start with all unique user IDs
        all_users = pd.concat([
            user_features[['user_id']] if not user_features.empty else pd.DataFrame(),
            session_features[['user_id']] if not session_features.empty else pd.DataFrame(),
            effectiveness_features[['user_id']] if not effectiveness_features.empty else pd.DataFrame()
        ]).drop_duplicates()
        
        if all_users.empty:
            return pd.DataFrame()
        
        # Merge features
        final_features = all_users
        
        if not user_features.empty:
            final_features = final_features.merge(user_features, on='user_id', how='left')
            
        if not session_features.empty:
            final_features = final_features.merge(session_features, on='user_id', how='left')
            
        if not effectiveness_features.empty:
            final_features = final_features.merge(effectiveness_features, on='user_id', how='left')
        
        # Fill missing values
        numeric_columns = final_features.select_dtypes(include=[np.number]).columns
        final_features[numeric_columns] = final_features[numeric_columns].fillna(0)
        
        final_features['fitness_level'] = final_features.get('fitness_level', 'beginner').fillna('beginner')
        final_features['age_group'] = final_features.get('age_group', 'unknown').fillna('unknown')
        
        # Add timestamp
        final_features['feature_timestamp'] = datetime.now()
        
        return final_features
    
    def save_features(self, features_df: pd.DataFrame):
        """Save features to DuckDB"""
        if features_df.empty:
            logger.warning("No features to save")
            return
        
        conn = duckdb.connect(str(self.duckdb_path))
        
        # Clear existing features
        conn.execute("DELETE FROM user_features")
        
        # Prepare data for insertion
        # Convert lists to strings for storage
        features_df = features_df.copy()
        for col in ['preferred_locations', 'equipment_used', 'tried_methods']:
            if col in features_df.columns:
                features_df[col] = features_df[col].apply(
                    lambda x: str(x) if isinstance(x, list) else str([])
                )
        
        # Insert new features
        conn.execute("INSERT INTO user_features SELECT * FROM features_df")
        
        conn.close()
        logger.info(f"Saved {len(features_df)} user features to DuckDB")

def create_sample_data():
    """Create sample data for testing"""
    logger.info("Creating sample data...")
    
    conn = sqlite3.connect(LOCAL_DB_PATH / "recoveredge.db")
    cursor = conn.cursor()
    
    # Sample users
    users = ['user1', 'user2', 'user3', 'user4', 'user5']
    
    # Create sample interactions
    for user in users:
        for i in range(np.random.randint(5, 20)):
            cursor.execute("""
            INSERT INTO user_interactions 
            (interaction_id, user_id, session_id, interaction_type, timestamp, 
             content, metadata, location, equipment_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"{user}_interaction_{i}",
                user,
                f"{user}_session_{i // 3}",
                "chat",
                datetime.now() - timedelta(days=np.random.randint(0, 30)),
                f"Sample query about recovery method {i}",
                '{"response_length": ' + str(np.random.randint(50, 200)) + '}',
                np.random.choice(['home', 'gym', 'office']),
                '["foam_roller", "mat"]'
            ))
    
    # Create sample sessions
    for user in users:
        for i in range(np.random.randint(3, 10)):
            cursor.execute("""
            INSERT INTO recovery_sessions
            (session_id, user_id, planned_duration, actual_duration,
             completed_methods, skipped_methods, location, equipment_used,
             satisfaction_rating, created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"{user}_session_{i}",
                user,
                np.random.randint(10, 30),
                np.random.randint(8, 35),
                '["Deep Breathing", "Foam Rolling"]',
                '[]',
                np.random.choice(['home', 'gym']),
                '["foam_roller"]',
                np.random.randint(3, 5),
                datetime.now() - timedelta(days=np.random.randint(0, 30)),
                datetime.now() - timedelta(days=np.random.randint(0, 29))
            ))
    
    # Create sample effectiveness ratings
    methods = ['Foam Rolling', 'Deep Breathing', 'Cold Therapy', 'Stretching']
    fitness_levels = ['beginner', 'intermediate', 'advanced']
    age_groups = ['18-25', '26-35', '36-45', '46+']
    
    for user in users:
        for method in np.random.choice(methods, size=np.random.randint(2, 4), replace=False):
            cursor.execute("""
            INSERT INTO method_effectiveness
            (method_name, user_id, effectiveness_rating, duration_completed,
             reported_benefits, side_effects, timestamp, 
             user_fitness_level, user_age_group)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                method,
                user,
                np.random.randint(3, 5),
                np.random.randint(5, 20),
                '["reduced_soreness", "improved_flexibility"]',
                '[]',
                datetime.now() - timedelta(days=np.random.randint(0, 30)),
                np.random.choice(fitness_levels),
                np.random.choice(age_groups)
            ))
    
    conn.commit()
    conn.close()
    logger.info("Sample data created")

if __name__ == "__main__":
    # Ensure directories exist
    LOCAL_DB_PATH.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    # Check if we need sample data
    if not (LOCAL_DB_PATH / "recoveredge.db").exists():
        logger.warning("Database not found. Creating sample data...")
        # Initialize database first (you'd need to run the main script once)
        create_sample_data()
    
    # Run feature engineering
    feature_eng = LocalFeatureEngineering()
    features = feature_eng.run_pipeline()
    
    if not features.empty:
        print("\nFeature Summary:")
        print(f"Total users: {len(features)}")
        print(f"\nFeature columns: {list(features.columns)}")
        print(f"\nSample features:")
        print(features.head())
        
        # Save to CSV for inspection
        features.to_csv(LOCAL_DATA_PATH / "user_features.csv", index=False)
        print(f"\nFeatures saved to {LOCAL_DATA_PATH / 'user_features.csv'}")
    else:
        print("No features generated. Check if there's data in the database.")