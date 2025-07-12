"""
RecoverEdge Local Implementation
Local alternative to Databricks with Spark, MLflow, and vector search
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from flask import Flask, request, jsonify
from flask_cors import CORS

# Database imports
import sqlite3
import duckdb

# Spark local mode
from pyspark.sql import SparkSession

# Vector search
import faiss
import chromadb
# from chromadb.config import Settings

# ML imports
import mlflow

# Training takes place in model_training_beta.py
# import mlflow.sklearn
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split

# LangChain imports
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
LOCAL_DATA_PATH = Path("./data")
LOCAL_MODELS_PATH = Path("./models")
LOCAL_MLRUNS_PATH = Path("./mlruns")
LOCAL_DB_PATH = Path("./databases")

# Create directories
for path in [LOCAL_DATA_PATH, LOCAL_MODELS_PATH, LOCAL_MLRUNS_PATH, LOCAL_DB_PATH]:
    path.mkdir(parents=True, exist_ok=True)

class LocalVectorStore:
    """Local vector store using Chroma or FAISS"""
    
    def __init__(self, store_type: str = "chroma"):
        self.store_type = store_type
        self.embeddings = OpenAIEmbeddings()
        
        if store_type == "chroma":
            self.client = chromadb.PersistentClient(path=str(LOCAL_DB_PATH / "chroma"))
            self.collection = self.client.get_or_create_collection(
                name="recovery_methods",
                metadata={"hnsw:space": "cosine"}
            )
        else:
            # Initialize FAISS
            self.dimension = 1536  # OpenAI embedding dimension
            self.index = None
            self.documents = []
            self.load_or_create_faiss()
    
    def load_or_create_faiss(self):
        """Load existing FAISS index or create new one"""
        faiss_path = LOCAL_DB_PATH / "faiss_index.pkl"
        if faiss_path.exists():
            with open(faiss_path, 'rb') as f:
                data = pickle.load(f)
                self.index = data['index']
                self.documents = data['documents']
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
    
    def save_faiss(self):
        """Save FAISS index to disk"""
        if self.store_type == "faiss":
            faiss_path = LOCAL_DB_PATH / "faiss_index.pkl"
            with open(faiss_path, 'wb') as f:
                pickle.dump({
                    'index': self.index,
                    'documents': self.documents
                }, f)
    
    def add_documents(self, documents: List[Dict]) -> bool:
        """Add documents to vector store"""
        try:
            if self.store_type == "chroma":
                # Prepare data for Chroma
                ids = [doc['id'] for doc in documents]
                texts = [doc['chunk_text'] for doc in documents]
                metadatas = [{
                    'title': doc.get('title', ''),
                    'category': doc.get('category', ''),
                    'topic': doc.get('topic', ''),
                    'chunk_index': doc.get('chunk_index', 0)
                } for doc in documents]
                
                # Generate embeddings
                embeddings = self.embeddings.embed_documents(texts)
                
                # Add to Chroma
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
            else:
                # FAISS implementation
                texts = [doc['chunk_text'] for doc in documents]
                embeddings = self.embeddings.embed_documents(texts)
                
                # Add to FAISS
                embeddings_array = np.array(embeddings).astype('float32')
                self.index.add(embeddings_array)
                self.documents.extend(documents)
                self.save_faiss()
            
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def search_similar(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search for similar documents"""
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            if self.store_type == "chroma":
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=num_results
                )
                
                # Format results
                documents = []
                for i in range(len(results['ids'][0])):
                    doc = {
                        'id': results['ids'][0][i],
                        'chunk_text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    }
                    documents.append(doc)
                return documents
            else:
                # FAISS search
                query_array = np.array([query_embedding]).astype('float32')
                distances, indices = self.index.search(query_array, num_results)
                
                # Get documents
                documents = []
                for idx, dist in zip(indices[0], distances[0]):
                    if idx < len(self.documents):
                        doc = self.documents[idx].copy()
                        doc['distance'] = float(dist)
                        documents.append(doc)
                return documents
                
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []

class LocalDataStore:
    """Local data storage using SQLite and DuckDB"""
    
    def __init__(self):
        self.sqlite_path = LOCAL_DB_PATH / "recoveredge.db"
        self.duckdb_path = LOCAL_DB_PATH / "analytics.duckdb"
        self.init_databases()
    
    def init_databases(self):
        """Initialize database schemas"""
        # SQLite for transactional data
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_interactions (
            interaction_id TEXT PRIMARY KEY,
            user_id TEXT,
            session_id TEXT,
            interaction_type TEXT,
            timestamp DATETIME,
            content TEXT,
            metadata TEXT,
            location TEXT,
            equipment_used TEXT
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS recovery_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
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
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS method_effectiveness (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            method_name TEXT,
            user_id TEXT,
            effectiveness_rating INTEGER,
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
        
        # DuckDB for analytics
        duck_conn = duckdb.connect(str(self.duckdb_path))
        
        # Create feature store table
        duck_conn.execute("""
        CREATE TABLE IF NOT EXISTS user_features (
            user_id VARCHAR PRIMARY KEY,
            total_interactions INTEGER,
            unique_sessions INTEGER,
            avg_response_length DOUBLE,
            total_sessions INTEGER,
            avg_session_duration DOUBLE,
            avg_satisfaction DOUBLE,
            avg_methods_completed DOUBLE,
            completion_efficiency DOUBLE,
            avg_method_effectiveness DOUBLE,
            fitness_level VARCHAR,
            age_group VARCHAR,
            preferred_locations_count INTEGER,
            equipment_variety INTEGER,
            feature_timestamp TIMESTAMP
        )
        """)
        
        duck_conn.close()
    
    def insert_interaction(self, interaction: Dict) -> bool:
        """Insert user interaction"""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO user_interactions 
            (interaction_id, user_id, session_id, interaction_type, timestamp, 
             content, metadata, location, equipment_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction['interaction_id'],
                interaction['user_id'],
                interaction['session_id'],
                interaction['interaction_type'],
                interaction['timestamp'],
                interaction['content'],
                json.dumps(interaction.get('metadata', {})),
                interaction.get('location', ''),
                json.dumps(interaction.get('equipment_used', []))
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error inserting interaction: {e}")
            return False
    
    def get_user_features(self, user_id: str) -> Dict:
        """Get user features from DuckDB"""
        try:
            conn = duckdb.connect(str(self.duckdb_path))
            
            result = conn.execute("""
            SELECT * FROM user_features 
            WHERE user_id = ? 
            ORDER BY feature_timestamp DESC 
            LIMIT 1
            """, [user_id]).fetchone()
            
            if result:
                columns = [desc[0] for desc in conn.description]
                return dict(zip(columns, result))
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting user features: {e}")
            return None

class LocalSparkProcessor:
    """Local Spark processing for feature engineering"""
    
    def __init__(self):
        # Initialize Spark in local mode
        self.spark = SparkSession.builder \
            .appName("RecoverEdgeLocal") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
    
    def process_user_features(self):
        """Process user features using Spark"""
        try:
            # Read from SQLite into Spark
            jdbc_url = f"jdbc:sqlite:{LOCAL_DB_PATH / 'recoveredge.db'}"
            
            # For demo, use pandas to read and convert to Spark
            # In production, use proper JDBC driver
            conn = sqlite3.connect(LOCAL_DB_PATH / "recoveredge.db")
            
            # Load data
            interactions_pd = pd.read_sql("SELECT * FROM user_interactions", conn)
            sessions_pd = pd.read_sql("SELECT * FROM recovery_sessions", conn)
            effectiveness_pd = pd.read_sql("SELECT * FROM method_effectiveness", conn)
            conn.close()
            
            # Convert to Spark DataFrames
            interactions_df = self.spark.createDataFrame(interactions_pd)
            sessions_df = self.spark.createDataFrame(sessions_pd)
            effectiveness_df = self.spark.createDataFrame(effectiveness_pd)
            
            # Register as temp views
            interactions_df.createOrReplaceTempView("user_interactions")
            sessions_df.createOrReplaceTempView("recovery_sessions")
            effectiveness_df.createOrReplaceTempView("method_effectiveness")
            
            # Feature engineering
            features_df = self.spark.sql("""
            WITH user_interaction_features AS (
                SELECT 
                    user_id,
                    COUNT(*) as total_interactions,
                    COUNT(DISTINCT session_id) as unique_sessions,
                    AVG(LENGTH(content)) as avg_response_length
                FROM user_interactions
                GROUP BY user_id
            ),
            session_features AS (
                SELECT 
                    user_id,
                    COUNT(*) as total_sessions,
                    AVG(actual_duration) as avg_session_duration,
                    AVG(satisfaction_rating) as avg_satisfaction
                FROM recovery_sessions
                GROUP BY user_id
            ),
            effectiveness_features AS (
                SELECT 
                    user_id,
                    AVG(effectiveness_rating) as avg_method_effectiveness,
                    MAX(user_fitness_level) as fitness_level,
                    MAX(user_age_group) as age_group
                FROM method_effectiveness
                GROUP BY user_id
            )
            SELECT 
                COALESCE(ui.user_id, s.user_id, e.user_id) as user_id,
                COALESCE(ui.total_interactions, 0) as total_interactions,
                COALESCE(ui.unique_sessions, 0) as unique_sessions,
                COALESCE(ui.avg_response_length, 0) as avg_response_length,
                COALESCE(s.total_sessions, 0) as total_sessions,
                COALESCE(s.avg_session_duration, 0) as avg_session_duration,
                COALESCE(s.avg_satisfaction, 3.0) as avg_satisfaction,
                COALESCE(e.avg_method_effectiveness, 3.0) as avg_method_effectiveness,
                COALESCE(e.fitness_level, 'beginner') as fitness_level,
                COALESCE(e.age_group, 'unknown') as age_group,
                CURRENT_TIMESTAMP as feature_timestamp
            FROM user_interaction_features ui
            FULL OUTER JOIN session_features s ON ui.user_id = s.user_id
            FULL OUTER JOIN effectiveness_features e ON ui.user_id = e.user_id
            """)
            
            # Convert to pandas and save to DuckDB
            features_pd = features_df.toPandas()
            
            duck_conn = duckdb.connect(str(self.duckdb_path))
            duck_conn.execute("DELETE FROM user_features")  # Clear old features
            duck_conn.execute("""
            INSERT INTO user_features SELECT * FROM features_pd
            """)
            duck_conn.close()
            
            logger.info(f"Processed features for {len(features_pd)} users")
            return True
            
        except Exception as e:
            logger.error(f"Error processing features: {e}")
            return False

class LocalMLflowModels:
    def __init__(self):
        mlflow.set_tracking_uri("./mlruns")
        self.recommendation_model = None
        self.load_models()
    
    def load_models(self):
        """Load the unified recommendation model"""
        try:
            # Load the complete model from model_training_beta.py
            model_uri = f"file://{LOCAL_MODELS_PATH}/combined_model"
            
            if Path(model_uri).exists():
                self.recommendation_model = mlflow.pyfunc.load_model(model_uri)
                logger.info("Loaded combined recommendation model")
            else:
                # Fallback: load individual components
                self._load_individual_models()
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.recommendation_model = None
    
    def _load_individual_models(self):
        """Fallback method to load individual model components"""
        try:
            # Load models and artifacts
            with open(LOCAL_MODELS_PATH / "effectiveness_model.pkl", 'rb') as f:
                effectiveness_model = pickle.load(f)
            
            with open(LOCAL_MODELS_PATH / "engagement_model.pkl", 'rb') as f:
                engagement_model = pickle.load(f)
            
            with open(LOCAL_MODELS_PATH / "model_artifacts.pkl", 'rb') as f:
                artifacts = pickle.load(f)
            
            # Create the same model class used in training
            from notebooks import model_training_beta as models
            
            self.recommendation_model = models.RecoveryRecommendationModel(
                effectiveness_model=effectiveness_model,
                engagement_model=engagement_model,
                encoders=artifacts['encoders'],
                scaler=artifacts['scaler']
            )
            
            logger.info("Loaded individual model components")
            
        except ImportError:
            logger.error("Could not import RecoveryRecommendationModel")
            self.recommendation_model = None
        except Exception as e:
            logger.error(f"Error loading individual models: {e}")
            self.recommendation_model = None
    
    def predict_recommendations(self, user_features: Dict) -> Dict:
        """Generate recommendations using the trained model"""
        if not self.recommendation_model:
            logger.warning("No model loaded, returning default recommendations")
            return self.get_default_recommendations()
        
        try:
            # Convert to DataFrame format expected by model
            input_df = pd.DataFrame([user_features])
            
            # Use the model's prediction method
            predictions = self.recommendation_model.predict(None, input_df)
            return predictions[0] if predictions else self.get_default_recommendations()
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return self.get_default_recommendations()
    
    def get_default_recommendations(self) -> Dict:
        """Enhanced default recommendations"""
        return {
            "recommended_methods": [
                {
                    "method": "Deep Breathing",
                    "confidence": 0.8,
                    "reason": "Safe and effective for all users",
                    "estimated_duration": 5,
                    "required_equipment": []
                },
                {
                    "method": "Gentle Stretching",
                    "confidence": 0.75,
                    "reason": "Improves flexibility and reduces tension",
                    "estimated_duration": 10,
                    "required_equipment": ["mat"]
                },
                {
                    "method": "Legs Up The Wall",
                    "confidence": 0.7,
                    "reason": "Passive recovery technique",
                    "estimated_duration": 10,
                    "required_equipment": ["wall"]
                }
            ],
            "effectiveness_score": 3.5,
            "engagement_probability": 0.6,
            "optimal_duration": 15,
            "model_version": "default_fallback",
            "user_profile": {
                "fitness_level": "unknown",
                "recommendation_basis": "general_population"
            }
        }

class LocalRAGSystem:
    """Complete local RAG system"""
    def __init__(self):

        self.vector_store = LocalVectorStore(store_type="chroma")
        self.data_store = LocalDataStore()
        self.spark_processor = LocalSparkProcessor()
        self.ml_models = LocalMLflowModels()
        
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        # Initialize system
        self.initialize()

    def check_model_availability(self):
        """Check if models are available, guide user if not"""
        required_files = [
            LOCAL_MODELS_PATH / "effectiveness_model.pkl",
            LOCAL_MODELS_PATH / "engagement_model.pkl",
            LOCAL_MODELS_PATH / "model_artifacts.pkl"
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            logger.warning("=" * 60)
            logger.warning("MISSING ML MODELS!")
            logger.warning("The following model files are missing:")
            for file in missing_files:
                logger.warning(f"  - {file}")
            logger.warning("")
            logger.warning("Please run the following commands to train models:")
            logger.warning("  1. python feature_engineering.py")
            logger.warning("  2. python model_training_beta.py")
            logger.warning("=" * 60)
            
            # Still allow server to run with default recommendations
            return False
        
        return True
    
    def initialize(self):
        """Initialize the local system"""
        logger.info("Initializing local RAG system...")
        
        # Load initial knowledge base
        self.load_initial_knowledge()

        # Check models (don't train here!)
        models_available = self.check_model_availability()
        if not models_available:
            logger.info("Server will use default recommendations until models are trained")
        
        logger.info("Local RAG system initialized")
    
    def load_initial_knowledge(self):
        """Load initial recovery knowledge"""
        knowledge_docs = [
            {
                "id": "foam_rolling_guide",
                "title": "Comprehensive Foam Rolling Guide",
                "content": """
                Foam rolling is a form of self-myofascial release that helps break up fascial adhesions and scar tissue.
                Benefits include increased blood flow, reduced muscle soreness, improved range of motion, and faster recovery.
                Technique: Apply moderate pressure and roll slowly (1 inch per second) over muscle groups.
                Duration: 30-60 seconds per muscle group, focusing on tight areas.
                Common areas: IT band, quadriceps, hamstrings, calves, back, and glutes.
                Research shows foam rolling can reduce DOMS by up to 30% when used consistently.
                """,
                "category": "self_massage",
                "topic": "foam_rolling",
                "tags": ["foam_rolling", "myofascial_release", "DOMS", "recovery"]
            },
            {
                "id": "cold_therapy_guide",
                "title": "Cold Therapy and Recovery Protocols",
                "content": """
                Cold therapy reduces inflammation and speeds recovery through vasoconstriction.
                Ice baths: 50-60°F (10-15°C) for 10-15 minutes post-workout.
                Cold showers: 2-3 minutes of cold water after regular shower.
                Benefits: Reduced inflammation, decreased muscle damage, improved recovery time.
                Research indicates 11-15 minutes of cold exposure weekly optimizes benefits.
                """,
                "category": "temperature",
                "topic": "cold_therapy",
                "tags": ["cold_therapy", "ice_bath", "inflammation", "recovery"]
            },
            {
                "id": "stretching_guide",
                "title": "Dynamic and Static Stretching for Recovery",
                "content": """
                Stretching improves flexibility, reduces muscle tension, and enhances recovery.
                Static stretching: Hold stretches for 30-60 seconds post-workout.
                Dynamic stretching: Controlled movements through full range of motion.
                Key areas: Hip flexors, hamstrings, quadriceps, calves, shoulders.
                Best performed when muscles are warm, ideally after light activity.
                """,
                "category": "flexibility",
                "topic": "stretching",
                "tags": ["stretching", "flexibility", "mobility", "recovery"]
            }
        ]
        
        # Check if documents already loaded
        existing_docs = self.vector_store.search_similar("foam rolling", num_results=1)
        if existing_docs:
            logger.info("Knowledge base already loaded")
            return
        
        # Process documents into chunks
        processed_docs = []
        for doc in knowledge_docs:
            chunks = self.text_splitter.split_text(doc["content"])
            
            for i, chunk in enumerate(chunks):
                processed_doc = doc.copy()
                processed_doc.update({
                    "chunk_text": chunk,
                    "chunk_index": i,
                    "id": f"{doc['id']}_chunk_{i}"
                })
                processed_docs.append(processed_doc)
        
        # Add to vector store
        self.vector_store.add_documents(processed_docs)
        logger.info(f"Loaded {len(processed_docs)} document chunks")
    
    def get_response(self, message: str, user_id: str = None, 
                    session_id: str = None) -> Tuple[str, List[str]]:
        """Get response using local RAG"""
        try:
            # Search similar chunks
            similar_chunks = self.vector_store.search_similar(message, num_results=3)
            
            if not similar_chunks:
                response = "I don't have specific information about that topic. Please consult with a healthcare professional."
                return response, []
            
            # Build context
            context = "\n\n".join([chunk.get("chunk_text", "") for chunk in similar_chunks])
            
            # Generate response
            prompt = f"""
            You are a knowledgeable recovery and fitness assistant. Use the following context to answer the question.
            Always provide evidence-based advice and remind users to consult healthcare professionals for medical concerns.
            
            Context: {context}
            
            Question: {message}
            
            Helpful Answer:"""
            
            response = self.llm.predict(prompt)
            
            # Log interaction if user_id provided
            if user_id and session_id:
                self.log_interaction(user_id, session_id, message, response, similar_chunks)
            
            chunk_ids = [chunk.get("id", "") for chunk in similar_chunks]
            return response, chunk_ids
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error. Please try again.", []
    
    def log_interaction(self, user_id: str, session_id: str, query: str,
                       response: str, retrieved_chunks: List[Dict]):
        """Log user interaction"""
        try:
            interaction = {
                "interaction_id": f"{session_id}_{datetime.now().timestamp()}",
                "user_id": user_id,
                "session_id": session_id,
                "interaction_type": "chat",
                "timestamp": datetime.now(),
                "content": json.dumps({
                    "query": query,
                    "response": response,
                    "retrieved_chunks": len(retrieved_chunks)
                }),
                "metadata": {
                    "response_length": len(response),
                    "chunks_retrieved": len(retrieved_chunks)
                }
            }
            
            self.data_store.insert_interaction(interaction)
            logger.info(f"Logged interaction for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error logging interaction: {e}")
    
    def get_recommendations(self, user_id: str, context: Dict = None) -> Dict:
        """Get personalized recommendations"""
        # Get user features
        user_features = self.data_store.get_user_features(user_id)
        
        if not user_features:
            # Create default features for new user
            user_features = {
                'total_interactions': 0,
                'unique_sessions': 0,
                'avg_response_length': 0,
                'total_sessions': 0,
                'avg_session_duration': 0,
                'avg_satisfaction': 3.0,
                'avg_methods_completed': 0,
                'completion_efficiency': 0,
                'avg_method_effectiveness': 3.0,
                'fitness_level': 'beginner',
                'age_group': 'unknown'
            }
        
        # Add context if provided
        if context:
            user_features.update({
                'current_location': context.get('location', 'home'),
                'available_time': context.get('available_time', 15),
                'available_equipment': context.get('available_equipment', [])
            })
        
        # Generate predictions
        recommendations = self.ml_models.predict_recommendations(user_features)
        
        # Add context-based filtering
        if context and context.get('available_time'):
            recommendations['optimal_duration'] = context['available_time']
            
            # Filter methods based on time
            if context['available_time'] <= 10:
                # Filter to quick methods
                quick_methods = [
                    m for m in recommendations['recommended_methods'] 
                    if m['method'] in ['Deep Breathing', 'Legs Up The Wall', 'Tennis Ball Foot Massage']
                ]
                if quick_methods:
                    recommendations['recommended_methods'] = quick_methods
        
        return recommendations
    
    def process_features_batch(self):
        """Process user features in batch using Spark"""
        logger.info("Starting batch feature processing...")
        success = self.spark_processor.process_user_features()
        if success:
            logger.info("Batch feature processing completed")
        else:
            logger.error("Batch feature processing failed")
        return success

# Initialize the local RAG system
rag_system = LocalRAGSystem()

# Flask API endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "RecoverEdge Local RAG Server",
        "components": {
            "vector_store": "chroma",
            "data_store": "sqlite/duckdb",
            "ml_framework": "mlflow_local",
            "spark": "local_mode"
        }
    }
    
    # Check component health
    try:
        # Test vector search
        rag_system.vector_store.search_similar("test", num_results=1)
        status["components"]["vector_store_status"] = "healthy"
    except:
        status["components"]["vector_store_status"] = "error"
        status["status"] = "degraded"
    
    return jsonify(status)

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint"""
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
        
        user_message = data['message']
        user_id = data.get('user_id', 'anonymous')
        session_id = data.get('session_id', 'default')
        
        response, chunk_ids = rag_system.get_response(user_message, user_id, session_id)
        
        return jsonify({
            "response": response,
            "session_id": session_id,
            "retrieved_chunks": len(chunk_ids),
            "powered_by": "local_rag",
            "error": None
        })
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return jsonify({
            "response": "I encountered an error processing your request.",
            "error": str(e)
        }), 500

@app.route('/recommendations/<user_id>', methods=['POST'])
def get_recommendations(user_id: str):
    """Get personalized recommendations"""
    try:
        data = request.json or {}
        context = {
            "location": data.get("currentLocation", "home"),
            "available_time": data.get("availableTime", 15),
            "available_equipment": data.get("availableEquipment", [])
        }
        
        recommendations = rag_system.get_recommendations(user_id, context)
        
        return jsonify(recommendations)
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analytics/insights', methods=['GET'])
def get_analytics_insights():
    """Get analytics insights using DuckDB"""
    try:
        conn = duckdb.connect(str(LOCAL_DB_PATH / "analytics.duckdb"))
        
        # Get user statistics
        user_stats = conn.execute("""
        SELECT 
            COUNT(*) as total_users,
            AVG(total_sessions) as avg_sessions_per_user,
            AVG(avg_satisfaction) as overall_satisfaction,
            AVG(avg_method_effectiveness) as overall_effectiveness
        FROM user_features
        """).fetchone()
        
        # Get popular recovery methods (from SQLite)
        sqlite_conn = sqlite3.connect(LOCAL_DB_PATH / "recoveredge.db")
        popular_methods = pd.read_sql("""
        SELECT 
            method_name,
            COUNT(*) as usage_count,
            AVG(effectiveness_rating) as avg_rating
        FROM method_effectiveness
        GROUP BY method_name
        ORDER BY usage_count DESC
        LIMIT 5
        """, sqlite_conn)
        sqlite_conn.close()
        
        insights = {
            "user_statistics": {
                "total_users": user_stats[0] if user_stats else 0,
                "avg_sessions_per_user": float(user_stats[1]) if user_stats else 0,
                "overall_satisfaction": float(user_stats[2]) if user_stats else 0,
                "overall_effectiveness": float(user_stats[3]) if user_stats else 0
            },
            "popular_methods": popular_methods.to_dict('records') if not popular_methods.empty else [],
            "system_info": {
                "storage_backend": "local",
                "vector_db": "chroma",
                "analytics_db": "duckdb",
                "ml_tracking": "mlflow_local"
            }
        }
        
        return jsonify(insights)
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/process_features', methods=['POST'])
def process_features():
    """Trigger batch feature processing"""
    try:
        success = rag_system.process_features_batch()
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Feature processing completed"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Feature processing failed"
            }), 500
            
    except Exception as e:
        logger.error(f"Error processing features: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("RecoverEdge Local RAG Server")
    print("Running with local alternatives to Databricks")
    print("=" * 60)
    print(f"Data directory: {LOCAL_DATA_PATH}")
    print(f"Models directory: {LOCAL_MODELS_PATH}")
    print(f"MLflow tracking: {LOCAL_MLRUNS_PATH}")
    print(f"Databases: {LOCAL_DB_PATH}")
    print("=" * 60)
    print("Server running on http://localhost:8000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8000, debug=True)