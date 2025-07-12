"""
Enhanced Feature Engineering Script with Equipment Validation
Incorporates realistic method-equipment relationships while maintaining full database integration
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
    """Local feature engineering pipeline with validated synthetic data generation"""
    
    def __init__(self):
        self.sqlite_path = LOCAL_DB_PATH / "recoveredge.db"
        self.duckdb_path = LOCAL_DB_PATH / "analytics.duckdb"
        
        # Recovery methods from Swift code
        self.recovery_methods = [
            "Legs Up The Wall",
            "Foam Rolling - Legs", 
            "Deep Breathing",
            "Cold Water Immersion",
            "Gentle Stretching",
            "Massage Gun Therapy",
            "Progressive Muscle Relaxation",
            "Tennis Ball Foot Massage",
            "Red Light Recovery Session",
            "Red Light Joint Therapy",
            "Sauna Recovery Session",
            "Cold Tank Immersion",
            "Hot Tank Therapy",
            "Contrast Hot-Cold Therapy",
            "Normatec Leg Recovery",
            "Normatec Arm Recovery",
            "Vibration Plate Recovery",
            "Vibration Plate Stretching",
            "Hypervolt Percussion Therapy",
            "Targeted Heat Therapy",
            "Shoulder Mobility Routine",
            "Band-Assisted Shoulder Stretches",
            "Quadriceps Stretch Routine",
            "Band-Assisted Quad Stretches",
            "Calf and Achilles Stretches",
            "Band-Assisted Calf Stretches",
            "Hamstring Flexibility Routine",
            "Band-Assisted Hamstring Stretches",
            "Glute and Hip Stretch Routine",
            "Band-Assisted Glute Stretches"
        ]
        
        self.locations = ["home", "gym", "hotel", "court"]
        
        self.equipment_options = [
            "Towel",
            "Wall",
            "Ground",
            "Water Bottle",
            "Pillow",
            "Chair",
            "Bench",
            "Stairs",
            "Court Wall",
            "Yoga Mat",
            "Resistance Bands",
            "Stretching Straps",
            "Foam Roller",
            "Massage Gun",
            "Lacrosse Ball",
            "Tennis Ball",
            "Ice Pack",
            "Bathtub",
            "Ice Bath",
            "Cold Tank",
            "Hot Tank",
            "Sauna",
            "Hot Pad",
            "Red Light Therapy",
            "Normatec",
            "Vibration Plate",
            "Hypervolt Gun"
        ]
        
        # Method-equipment mapping for validation
        self.method_equipment_mapping = {
            "Legs Up The Wall": ["Wall"],
            "Foam Rolling - Legs": ["Foam Roller"],
            "Deep Breathing": [],  # No equipment needed
            "Cold Water Immersion": ["Bathtub", "Ice Bath"],  # Either one works
            "Gentle Stretching": ["Yoga Mat"],
            "Massage Gun Therapy": ["Massage Gun"],
            "Progressive Muscle Relaxation": [],  # No equipment needed
            "Tennis Ball Foot Massage": ["Tennis Ball"],
            "Red Light Recovery Session": ["Red Light Therapy"],
            "Red Light Joint Therapy": ["Red Light Therapy"],
            "Sauna Recovery Session": ["Sauna"],
            "Cold Tank Immersion": ["Cold Tank"],
            "Hot Tank Therapy": ["Hot Tank"],
            "Contrast Hot-Cold Therapy": ["Hot Tank", "Cold Tank"],  # Needs both
            "Normatec Leg Recovery": ["Normatec"],
            "Normatec Arm Recovery": ["Normatec"],
            "Vibration Plate Recovery": ["Vibration Plate"],
            "Vibration Plate Stretching": ["Vibration Plate"],
            "Hypervolt Percussion Therapy": ["Hypervolt Gun"],
            "Targeted Heat Therapy": ["Hot Pad"],
            "Shoulder Mobility Routine": [],  # No equipment needed
            "Band-Assisted Shoulder Stretches": ["Resistance Bands"],
            "Quadriceps Stretch Routine": [],  # No equipment needed
            "Band-Assisted Quad Stretches": ["Resistance Bands"],
            "Calf and Achilles Stretches": ["Wall"],
            "Band-Assisted Calf Stretches": ["Resistance Bands"],
            "Hamstring Flexibility Routine": [],  # No equipment needed
            "Band-Assisted Hamstring Stretches": ["Resistance Bands"],
            "Glute and Hip Stretch Routine": [],  # No equipment needed
            "Band-Assisted Glute Stretches": ["Resistance Bands"]
        }
        
        # Location-appropriate equipment (realistic availability)
        self.location_equipment_mapping = {
            "gym": 
            [
            "Towel", "Wall", "Ground", "Water Bottle", "Pillow",
            "Chair", "Bench", "Stairs", "Court Wall", "Yoga Mat",
            "Resistance Bands", "Stretching Straps", "Foam Roller",
            "Massage Gun", "Lacrosse Ball", "Tennis Ball", "Ice Pack",
            "Bathtub", "Ice Bath", "Cold Tank", "Hot Tank", "Sauna",
            "Hot Pad", "Red Light Therapy", "Normatec", "Vibration Plate",
            "Hypervolt Gun"
            ]
            ,
            "home": 
            [
            "Towel", "Wall", "Ground", "Water Bottle", "Pillow",
            "Chair", "Bench", "Stairs", "Court Wall", "Yoga Mat",
            "Resistance Bands", "Stretching Straps", "Foam Roller",
            "Massage Gun", "Lacrosse Ball", "Tennis Ball", "Ice Pack",
            "Bathtub", "Ice Bath", "Cold Tank", "Hot Tank", "Sauna",
            "Hot Pad", "Red Light Therapy", "Normatec", "Vibration Plate",
            "Hypervolt Gun"
        ]
        
            ,
            "hotel": 
            [
            "Towel", "Wall", "Ground", "Water Bottle", "Pillow",
            "Chair", "Bench", "Stairs", "Court Wall", "Yoga Mat",
            "Resistance Bands", "Stretching Straps", "Foam Roller",
            "Massage Gun", "Lacrosse Ball", "Tennis Ball", "Ice Pack",
            "Bathtub", "Ice Bath", "Cold Tank", "Hot Tank", "Sauna",
            "Hot Pad", "Red Light Therapy", "Normatec", "Vibration Plate",
            "Hypervolt Gun"
        ]
            ,
            "court": 
            [
                "Towel", "Wall", "Ground", "Water Bottle", "Pillow",
            "Chair", "Bench", "Stairs", "Court Wall", "Yoga Mat",
            "Resistance Bands", "Stretching Straps", "Foam Roller",
            "Massage Gun", "Lacrosse Ball", "Tennis Ball", "Ice Pack",
            "Bathtub", "Ice Bath", "Cold Tank", "Hot Tank", "Sauna",
            "Hot Pad", "Red Light Therapy", "Normatec", "Vibration Plate",
            "Hypervolt Gun"
        ]
        
        }
    
    def get_available_methods_for_equipment(self, available_equipment):
        """Get methods that can be performed with available equipment"""
        available_methods = []
        
        for method, required_equipment in self.method_equipment_mapping.items():
            if not required_equipment:  # No equipment needed
                available_methods.append(method)
            elif len(required_equipment) == 1:  # Single equipment needed
                if required_equipment[0] in available_equipment:
                    available_methods.append(method)
            else:  # Multiple equipment needed (like Contrast Therapy)
                if all(eq in available_equipment for eq in required_equipment):
                    available_methods.append(method)
        
        return available_methods
    
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
        
        # Create tables in DuckDB
        duckdb_conn = duckdb.connect(str(self.duckdb_path))
        
        # Create user_features table
        duckdb_conn.execute("DROP TABLE IF EXISTS user_features")
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
        
        # Create analytics tables in DuckDB for better querying
        duckdb_conn.execute("DROP TABLE IF EXISTS session_analytics")
        duckdb_conn.execute("""
        CREATE TABLE session_analytics (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            location TEXT,
            equipment_count INTEGER,
            method_count INTEGER,
            completion_rate REAL,
            satisfaction_rating INTEGER,
            session_duration INTEGER,
            created_at DATETIME
        )
        """)
        
        duckdb_conn.execute("DROP TABLE IF EXISTS method_analytics")
        duckdb_conn.execute("""
        CREATE TABLE method_analytics (
            method_name TEXT,
            total_uses INTEGER,
            avg_effectiveness REAL,
            avg_satisfaction REAL,
            required_equipment TEXT,
            difficulty_level INTEGER
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
        
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_interactions")
        result = cursor.fetchone()
        interaction_users = result[0] if result else 0
        
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM recovery_sessions")
        result = cursor.fetchone()
        session_users = result[0] if result else 0
        
        conn.close()
        
        return max(interaction_users, session_users) >= min_users
    
    def generate_synthetic_data(self, n_users=200, days_back=90):
        """Generate comprehensive synthetic raw data with equipment validation"""
        logger.info(f"Generating validated synthetic data for {n_users} users over {days_back} days...")
        
        # Clear existing data
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_interactions")
        cursor.execute("DELETE FROM recovery_sessions")
        cursor.execute("DELETE FROM method_effectiveness")
        conn.commit()
        
        # Generate user profiles with realistic equipment
        np.random.seed(42)
        user_profiles = self._generate_realistic_user_profiles(n_users)
        
        # Generate data for each user
        for user_id, profile in user_profiles.items():
            self._generate_user_interactions(cursor, user_id, profile, days_back)
            self._generate_validated_recovery_sessions(cursor, user_id, profile, days_back)
            self._generate_method_effectiveness(cursor, user_id, profile, days_back)
        
        conn.commit()
        conn.close()
        
        # Validate generated data
        self._validate_generated_data()
        
        logger.info(f"Validated synthetic data generation completed for {n_users} users")
    
    def _generate_realistic_user_profiles(self, n_users):
        """Generate realistic user profiles with proper equipment-method relationships"""
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
                equipment_budget = 'low'
            elif fitness_level == 'intermediate':
                activity_level = np.random.uniform(0.6, 0.9)
                completion_tendency = np.random.uniform(0.7, 0.95)
                equipment_budget = 'medium'
            else:  # advanced
                activity_level = np.random.uniform(0.8, 1.0)
                completion_tendency = np.random.uniform(0.8, 1.0)
                equipment_budget = 'high'
            
            # Generate realistic equipment based on preferred locations and budget
            preferred_locations = np.random.choice(
                self.locations, 
                size=np.random.randint(1, 3), 
                replace=False
            ).tolist()
            
            # Build available equipment based on budget and location
            available_equipment = set()
            for location in preferred_locations:
                location_equipment = self.location_equipment_mapping[location]
                
                if equipment_budget == 'low':
                    # Basic equipment only
                    basic_equipment = [eq for eq in location_equipment if eq in [
                        "Towel", "Wall", "Ground", "Water Bottle", "Yoga Mat",
                        "Resistance Bands", "Tennis Ball", "Ice Pack"
                    ]]
                    if basic_equipment:
                        available_equipment.update(
                            np.random.choice(basic_equipment, 
                                           size=min(len(basic_equipment), np.random.randint(2, 5)),
                                           replace=False)
                        )
                elif equipment_budget == 'medium':
                    num_items = min(len(location_equipment), np.random.randint(4, 8))
                    available_equipment.update(
                        np.random.choice(location_equipment, size=num_items, replace=False)
                    )
                else:  # high budget
                    num_items = min(len(location_equipment), np.random.randint(6, len(location_equipment)))
                    available_equipment.update(
                        np.random.choice(location_equipment, size=num_items, replace=False)
                    )
            
            available_equipment = list(available_equipment)
            
            # Generate preferred methods based on available equipment
            available_methods = self.get_available_methods_for_equipment(available_equipment)
            
            if available_methods:
                num_preferred = min(len(available_methods), np.random.randint(3, 8))
                preferred_methods = np.random.choice(available_methods,
                                                   size=num_preferred,
                                                   replace=False).tolist()
            else:
                # Fallback to equipment-free methods
                equipment_free_methods = [method for method, equipment in self.method_equipment_mapping.items() 
                                        if not equipment]
                preferred_methods = np.random.choice(equipment_free_methods,
                                                   size=min(len(equipment_free_methods), 3),
                                                   replace=False).tolist()
            
            profiles[user_id] = {
                'fitness_level': fitness_level,
                'age_group': age_group,
                'activity_level': activity_level,
                'completion_tendency': completion_tendency,
                'equipment_budget': equipment_budget,
                'preferred_locations': preferred_locations,
                'available_equipment': available_equipment,
                'preferred_methods': preferred_methods
            }
        
        return profiles
    
    def _generate_user_interactions(self, cursor, user_id, profile, days_back):
        """Generate realistic user interactions"""
        n_interactions = int(np.random.poisson(profile['activity_level'] * 15))
        
        session_counter = 0
        for i in range(n_interactions):
            if i % np.random.randint(2, 5) == 0:
                session_counter += 1
            
            session_id = f"{user_id}_session_{session_counter:03d}"
            
            interaction_types = ['chat', 'query', 'feedback', 'request']
            interaction_type = np.random.choice(interaction_types, p=[0.4, 0.3, 0.2, 0.1])
            
            # Generate content based on interaction type and available methods
            if interaction_type == 'chat':
                content = f"How do I improve my {np.random.choice(['flexibility', 'recovery', 'soreness', 'performance'])}?"
            elif interaction_type == 'query':
                method = np.random.choice(profile['preferred_methods'])
                content = f"Tell me more about {method} for recovery"
            elif interaction_type == 'feedback':
                method = np.random.choice(profile['preferred_methods'])
                content = f"The {method} session was helpful"
            else:  # request
                location = np.random.choice(profile['preferred_locations'])
                content = f"Can you recommend recovery methods for {location}?"
            
            response_length = int(np.random.normal(100, 30))
            if response_length < 20:
                response_length = 20
            
            metadata = json.dumps({
                'response_length': response_length,
                'satisfaction': np.random.uniform(3, 5),
                'helpful': str(np.random.choice([True, False], p=[0.8, 0.2]))
            })
            
            location = np.random.choice(profile['preferred_locations'])
            
            # Only use equipment that's actually available at the location
            location_equipment = set(self.location_equipment_mapping[location])
            user_equipment = set(profile['available_equipment'])
            available_at_location = list(location_equipment.intersection(user_equipment))
            
            if available_at_location:
                equipment_used = json.dumps(
                    np.random.choice(available_at_location, 
                                   size=min(len(available_at_location), np.random.randint(1, 3)), 
                                   replace=False).tolist()
                )
            else:
                equipment_used = json.dumps([])
            
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
    
    def _generate_validated_recovery_sessions(self, cursor, user_id, profile, days_back):
        """Generate recovery sessions with proper method-equipment validation"""
        n_sessions = int(np.random.poisson(profile['activity_level'] * 10))
        
        for i in range(n_sessions):
            session_id = f"{user_id}_session_{i:03d}"
            planned_duration = np.random.randint(10, 45)
            
            # Select location
            location = np.random.choice(profile['preferred_locations'])
            
            # Get equipment available at this location for this user
            location_equipment = set(self.location_equipment_mapping[location])
            user_equipment = set(profile['available_equipment'])
            session_equipment = list(location_equipment.intersection(user_equipment))
            
            # Get methods that can actually be performed with available equipment
            available_methods = self.get_available_methods_for_equipment(session_equipment)
            
            # Filter to user's preferred methods that are doable
            doable_preferred_methods = [m for m in profile['preferred_methods'] 
                                      if m in available_methods]
            
            if not doable_preferred_methods:
                # Fallback to any available methods
                doable_preferred_methods = available_methods[:3] if available_methods else ["Deep Breathing"]
            
            # Select methods for this session
            n_methods = min(len(doable_preferred_methods), np.random.randint(1, 4))
            selected_methods = np.random.choice(doable_preferred_methods,
                                              size=n_methods,
                                              replace=False).tolist()
            
            # Determine which equipment is actually used
            equipment_used = set()
            for method in selected_methods:
                required_equipment = self.method_equipment_mapping.get(method, [])
                for eq in required_equipment:
                    if eq in session_equipment:
                        equipment_used.add(eq)
            
            # Actual duration and completion based on tendency
            if np.random.random() < profile['completion_tendency']:
                actual_duration = int(planned_duration * np.random.uniform(0.8, 1.2))
                completed_methods = selected_methods
                skipped_methods = []
            else:
                actual_duration = int(planned_duration * np.random.uniform(0.3, 0.8))
                n_completed = np.random.randint(0, len(selected_methods))
                completed_methods = selected_methods[:n_completed]
                skipped_methods = selected_methods[n_completed:]
            
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
                json.dumps(list(equipment_used)),
                satisfaction_rating,
                created_at,
                completed_at
            ))
    
    def _generate_method_effectiveness(self, cursor, user_id, profile, days_back):
        """Generate method effectiveness ratings"""
        for method in profile['preferred_methods']:
            n_ratings = np.random.randint(1, 5)
            
            for rating_idx in range(n_ratings):
                # Effectiveness rating influenced by fitness level
                if profile['fitness_level'] == 'advanced':
                    effectiveness_rating = np.random.uniform(3.5, 5.0)
                elif profile['fitness_level'] == 'intermediate':
                    effectiveness_rating = np.random.uniform(3.0, 4.5)
                else:  # beginner
                    effectiveness_rating = np.random.uniform(2.5, 4.0)
                
                duration_completed = np.random.randint(5, 30)
                
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
                
                side_effects = json.dumps([]) if np.random.random() > 0.1 else json.dumps(["mild_discomfort"])
                
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
    
    def _validate_generated_data(self):
        """Validate that all generated sessions have proper method-equipment relationships"""
        sessions_df = self.load_recovery_sessions()
        
        if sessions_df.empty:
            logger.warning("No sessions to validate")
            return
        
        validation_errors = []
        
        for idx, row in sessions_df.iterrows():
            completed_methods = row['completed_methods']
            equipment_used = row['equipment_used']
            
            for method in completed_methods:
                required_equipment = self.method_equipment_mapping.get(method, [])
                missing_equipment = [eq for eq in required_equipment if eq not in equipment_used]
                
                if missing_equipment:
                    validation_errors.append({
                        'session_id': row['session_id'],
                        'method': method,
                        'missing_equipment': missing_equipment,
                        'available_equipment': equipment_used
                    })
        
        if validation_errors:
            logger.error(f"Found {len(validation_errors)} validation errors in generated data!")
            for error in validation_errors[:5]:
                logger.error(f"  Session {error['session_id']}: {error['method']} missing {error['missing_equipment']}")
        else:
            logger.info("✅ All generated sessions have valid method-equipment relationships!")
    
    def run_pipeline(self, force_regenerate=False):
        """Run the complete feature engineering pipeline"""
        logger.info("Starting feature engineering pipeline...")
        
        # Initialize database
        self.initialize_database()
        
        # Check if we need to generate synthetic data
        if force_regenerate or not self.check_data_exists():
            logger.info("Generating validated synthetic data...")
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
        
        # Step 5: Create analytics tables in DuckDB
        self.create_analytics_tables(sessions_df, effectiveness_df)
        
        logger.info(f"Feature engineering completed for {len(final_features)} users")
        
        return final_features
    
    def create_analytics_tables(self, sessions_df, effectiveness_df):
        """Create analytics tables in DuckDB for better querying"""
        conn = duckdb.connect(str(self.duckdb_path))
        
        # Create session analytics
        if not sessions_df.empty:
            session_analytics = sessions_df.copy()
            session_analytics['method_count'] = session_analytics['completed_methods'].apply(len)
            session_analytics['equipment_count'] = session_analytics['equipment_used'].apply(len)
            session_analytics['completion_rate'] = session_analytics.apply(
                lambda x: len(x['completed_methods']) / (len(x['completed_methods']) + len(x['skipped_methods']))
                if (len(x['completed_methods']) + len(x['skipped_methods'])) > 0 else 0,
                axis=1
            )
            
            # Select relevant columns
            session_analytics = session_analytics[[
                'session_id', 'user_id', 'location', 'equipment_count',
                'method_count', 'completion_rate', 'satisfaction_rating',
                'actual_duration', 'created_at'
            ]].rename(columns={'actual_duration': 'session_duration'})
            
            # Clear and insert
            conn.execute("DELETE FROM session_analytics")
            conn.register('session_analytics_df', session_analytics)
            conn.execute("INSERT INTO session_analytics SELECT * FROM session_analytics_df")
            logger.info(f"Saved {len(session_analytics)} session analytics records")
        
        # Create method analytics
        if not effectiveness_df.empty:
            # Calculate method statistics
            method_stats = effectiveness_df.groupby('method_name').agg({
                'effectiveness_rating': ['count', 'mean'],
                'user_id': 'nunique'
            }).reset_index()
            
            method_stats.columns = ['method_name', 'total_uses', 'avg_effectiveness', 'unique_users']
            
            # Add satisfaction data from sessions
            if not sessions_df.empty:
                # Create method-satisfaction mapping
                method_satisfaction = []
                for _, row in sessions_df.iterrows():
                    for method in row['completed_methods']:
                        method_satisfaction.append({
                            'method_name': method,
                            'satisfaction_rating': row['satisfaction_rating']
                        })
                
                if method_satisfaction:
                    method_sat_df = pd.DataFrame(method_satisfaction)
                    avg_satisfaction = method_sat_df.groupby('method_name')['satisfaction_rating'].mean().reset_index()
                    avg_satisfaction.columns = ['method_name', 'avg_satisfaction']
                    
                    method_stats = method_stats.merge(avg_satisfaction, on='method_name', how='left')
                else:
                    method_stats['avg_satisfaction'] = 3.0
            else:
                method_stats['avg_satisfaction'] = 3.0
            
            # Add required equipment and difficulty
            method_stats['required_equipment'] = method_stats['method_name'].apply(
                lambda x: json.dumps(self.method_equipment_mapping.get(x, []))
            )
            
            # Assign difficulty levels based on equipment and method type
            method_stats['difficulty_level'] = method_stats['method_name'].apply(
                lambda x: self._get_method_difficulty(x)
            )
            
            # Select final columns
            method_analytics = method_stats[[
                'method_name', 'total_uses', 'avg_effectiveness', 
                'avg_satisfaction', 'required_equipment', 'difficulty_level'
            ]]
            
            # Clear and insert
            conn.execute("DELETE FROM method_analytics")
            conn.register('method_analytics_df', method_analytics)
            conn.execute("INSERT INTO method_analytics SELECT * FROM method_analytics_df")
            logger.info(f"Saved {len(method_analytics)} method analytics records")
        
        conn.close()
    
    def _get_method_difficulty(self, method_name):
        """Assign difficulty level to methods"""
        # High difficulty methods
        if any(term in method_name for term in ['Cold Tank', 'Ice Bath', 'Contrast', 'Advanced']):
            return 3
        # Medium difficulty methods
        elif any(term in method_name for term in ['Foam Rolling', 'Band-Assisted', 'Massage Gun', 
                                                  'Vibration', 'Hypervolt', 'Normatec']):
            return 2
        # Low difficulty methods
        else:
            return 1
    
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
        
        # Insert features
        try:
            conn.register('features_df', features_df)
            conn.execute("INSERT INTO user_features SELECT * FROM features_df")
            logger.info(f"Successfully saved {len(features_df)} user features to DuckDB")
        except Exception as e:
            logger.error(f"Error inserting features: {e}")
            # Save to CSV as backup
            features_df.to_csv(LOCAL_DATA_PATH / "user_features_backup.csv", index=False)
            logger.info(f"Saved features backup to {LOCAL_DATA_PATH / 'user_features_backup.csv'}")
        finally:
            conn.close()
    
    def query_analytics(self, query: str) -> pd.DataFrame:
        """Execute analytics queries on DuckDB"""
        conn = duckdb.connect(str(self.duckdb_path))
        try:
            result = conn.execute(query).df()
            return result
        finally:
            conn.close()
    
    def get_method_recommendations(self, user_id: str, location: str, available_time: int) -> list:
        """Get personalized method recommendations using DuckDB analytics"""
        conn = duckdb.connect(str(self.duckdb_path))
        
        try:
            # Get user features
            user_features = conn.execute(f"""
                SELECT * FROM user_features WHERE user_id = '{user_id}'
            """).df()
            
            if user_features.empty:
                logger.warning(f"No features found for user {user_id}")
                return []
            
            # Get user's tried methods
            tried_methods = json.loads(user_features.iloc[0]['tried_methods'])
            
            # Get method analytics
            method_stats = conn.execute("""
                SELECT * FROM method_analytics
                WHERE avg_effectiveness > 3.0
                AND avg_satisfaction > 3.0
                ORDER BY avg_effectiveness DESC, total_uses DESC
            """).df()
            
            recommendations = []
            
            for _, method in method_stats.iterrows():
                method_name = method['method_name']
                required_equipment = json.loads(method['required_equipment'])
                
                # Check if user has tried this method
                confidence = 0.8 if method_name in tried_methods else 0.6
                
                # Check equipment availability at location
                location_equipment = self.location_equipment_mapping.get(location, [])
                can_perform = all(eq in location_equipment for eq in required_equipment)
                
                if can_perform:
                    recommendations.append({
                        'method': method_name,
                        'confidence': confidence,
                        'reason': f"High effectiveness ({method['avg_effectiveness']:.1f}/5.0)",
                        'required_equipment': required_equipment,
                        'difficulty': method['difficulty_level']
                    })
            
            return recommendations[:5]  # Return top 5 recommendations
            
        finally:
            conn.close()


# Analytics helper functions
def print_analytics_summary(feature_eng: LocalFeatureEngineering):
    """Print comprehensive analytics summary using DuckDB queries"""
    print("\n=== ANALYTICS SUMMARY ===")
    
    # User engagement metrics
    engagement_stats = feature_eng.query_analytics("""
        SELECT 
            COUNT(DISTINCT user_id) as total_users,
            AVG(total_sessions) as avg_sessions_per_user,
            AVG(avg_satisfaction) as overall_satisfaction,
            AVG(completion_efficiency) as overall_completion_rate
        FROM user_features
    """)
    
    print("\nUser Engagement Metrics:")
    print(engagement_stats)
    
    # Method popularity
    method_popularity = feature_eng.query_analytics("""
        SELECT 
            method_name,
            total_uses,
            avg_effectiveness,
            avg_satisfaction,
            difficulty_level
        FROM method_analytics
        ORDER BY total_uses DESC
        LIMIT 10
    """)
    
    print("\nTop 10 Most Popular Methods:")
    print(method_popularity)
    
    # Location-based analytics
    location_stats = feature_eng.query_analytics("""
        SELECT 
            location,
            COUNT(*) as session_count,
            AVG(session_duration) as avg_duration,
            AVG(satisfaction_rating) as avg_satisfaction,
            AVG(completion_rate) as avg_completion
        FROM session_analytics
        GROUP BY location
        ORDER BY session_count DESC
    """)
    
    print("\nLocation-Based Analytics:")
    print(location_stats)
    
    # Equipment utilization
    print("\nEquipment-Method Relationships:")
    equipment_free_methods = [m for m, eq in feature_eng.method_equipment_mapping.items() if not eq]
    print(f"Equipment-free methods: {len(equipment_free_methods)}")
    print(f"Methods requiring equipment: {len(feature_eng.method_equipment_mapping) - len(equipment_free_methods)}")


if __name__ == "__main__":
    # Ensure directories exist
    LOCAL_DB_PATH.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    # Run enhanced feature engineering with validation
    feature_eng = LocalFeatureEngineering()
    features = feature_eng.run_pipeline(force_regenerate=True)
    
    if not features.empty:
        print("\nFeature Summary:")
        print(f"Total users: {len(features)}")
        print(f"\nFeature columns ({len(features.columns)}): {list(features.columns)}")
        print(f"\nSample features:")
        print(features.head())
        
        # Save to CSV for inspection
        features.to_csv(LOCAL_DATA_PATH / "user_features.csv", index=False)
        print(f"\nFeatures saved to {LOCAL_DATA_PATH / 'user_features.csv'}")
        
        # Print analytics summary
        print_analytics_summary(feature_eng)
        
        # Test personalized recommendations
        print("\n=== TESTING PERSONALIZED RECOMMENDATIONS ===")
        test_user = features.iloc[0]['user_id']
        recommendations = feature_eng.get_method_recommendations(test_user, 'gym', 30)
        print(f"\nRecommendations for {test_user} at gym with 30 minutes:")
        for rec in recommendations:
            print(f"  - {rec['method']} (confidence: {rec['confidence']:.1f})")
            print(f"    Reason: {rec['reason']}")
            print(f"    Equipment: {rec['required_equipment']}")
    else:
        print("No features generated. Check database initialization.")