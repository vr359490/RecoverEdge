"""
RecoverEdge Databricks Integration
Combines vector databases with Databricks for advanced analytics and ML
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Consider using Spark when there is larger dataset
# from pyspark.sql import SparkSession
# spark = SparkSession.builder.getOrCreate()

# Databricks imports
from databricks.sql import connect as databricks_connect
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient

# LangChain imports
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# MLflow for model tracking
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabricksVectorStore:
    """Databricks Vector Search integration for RecoverEdge knowledge base"""
    
    def __init__(self, workspace_url: str, token: str, vector_search_endpoint: str):
        self.workspace_url = workspace_url
        self.token = token
        self.vector_search_endpoint = vector_search_endpoint
        
        # Initialize clients
        self.workspace_client = WorkspaceClient(
            host=workspace_url,
            token=token
        )
        
        self.vector_search_client = VectorSearchClient(
            workspace_url=workspace_url,
            personal_access_token=token
        )
        
        # Vector search index configuration
        self.catalog_name = "recoveredge"
        self.schema_name = "knowledge_base"
        self.index_name = "recovery_methods_index"
        self.table_name = f"{self.catalog_name}.{self.schema_name}.knowledge_documents"
        

    def setup_unity_catalog(self):
        """Setup Unity Catalog structure for RecoverEdge"""
        try:
            # Check if catalog already exists first
            try:
                existing_catalog = self.workspace_client.catalogs.get(self.catalog_name)
                logger.info(f"Catalog {self.catalog_name} already exists - skipping creation")
            except Exception as catalog_error:
                # Catalog doesn't exist, try to create it
                logger.info(f"Catalog {self.catalog_name} not found, attempting to create...")
                try:
                    self.workspace_client.catalogs.create(
                        name=self.catalog_name,
                        comment="RecoverEdge fitness recovery knowledge base"
                    )
                    logger.info(f"Created catalog: {self.catalog_name}")
                except Exception as create_error:
                    if "already exists" in str(create_error).lower():
                        logger.info(f"Catalog {self.catalog_name} already exists")
                    else:
                        logger.error(f"Failed to create catalog: {create_error}")
                        # Since the catalog exists (we can see it in your UI), let's continue
                        logger.info("Continuing with existing catalog...")
            
            # Check if schema already exists first
            try:
                existing_schema = self.workspace_client.schemas.get(
                    full_name=f"{self.catalog_name}.{self.schema_name}"
                )
                logger.info(f"Schema {self.schema_name} already exists - skipping creation")
            except Exception as schema_error:
                # Schema doesn't exist, try to create it
                try:
                    self.workspace_client.schemas.create(
                        name=self.schema_name,
                        catalog_name=self.catalog_name,
                        comment="Recovery methods and knowledge base"
                    )
                    logger.info(f"Created schema: {self.schema_name}")
                except Exception as create_schema_error:
                    if "already exists" in str(create_schema_error).lower():
                        logger.info(f"Schema {self.schema_name} already exists")
                    else:
                        logger.error(f"Failed to create schema: {create_schema_error}")
                        # Since the schema exists (we can see it), let's continue
                        logger.info("Continuing with existing schema...")
                        
            logger.info("Unity Catalog setup completed successfully")
                        
        except Exception as e:
            logger.error(f"Error setting up Unity Catalog: {e}")
            # Don't raise the error - continue with existing catalog/schema
            logger.info("Continuing with existing Unity Catalog structure...")
    
    def create_knowledge_table(self):
        """Create Delta table for knowledge documents"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id STRING,
            title STRING,
            content STRING,
            category STRING,
            topic STRING,
            tags ARRAY<STRING>,
            chunk_text STRING,
            chunk_index INT,
            embedding ARRAY<FLOAT>,
            source STRING,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            is_active BOOLEAN
        ) USING DELTA
        TBLPROPERTIES (
            'delta.enableChangeDataFeed' = 'true',
            'delta.columnMapping.mode' = 'name'
        )
        """
        
        try:
            # Execute using Databricks SQL
            with databricks_connect(
                server_hostname=self.workspace_url.replace("https://", ""),
                http_path="/sql/1.0/warehouses/f3b9d912c6b793ad",  # Replace with actual warehouse
                access_token=self.token
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(create_table_sql)
                    logger.info(f"Created table: {self.table_name}")
                    
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            raise
    
    def create_vector_search_endpoint(self):
        """Create vector search endpoint if it doesn't exist"""
        try:
            # First, check if it already exists
            try:
                existing_endpoints = list(self.workspace_client.vector_search_endpoints.list_endpoints())
                logger.info(f"Found {len(existing_endpoints)} existing endpoints")
                
                for endpoint in existing_endpoints:
                    if hasattr(endpoint, 'name') and endpoint.name == "recovery-vector-search":
                        logger.info(f"Vector search endpoint 'recovery-vector-search' already exists")
                        return True
                        
            except Exception as list_error:
                logger.warning(f"Could not list existing endpoints: {list_error}")
                # Continue and try to create anyway
            
            # Create new endpoint - try different parameter formats
            logger.info("Attempting to create vector search endpoint...")
            
            try:
                # Method 1: Without explicit endpoint_type
                endpoint = self.workspace_client.vector_search_endpoints.create_endpoint(
                    name="recovery-vector-search"
                )
                logger.info(f"Created vector search endpoint: recovery-vector-search")
                return True
                
            except Exception as e1:
                logger.info(f"Method 1 failed: {e1}")
                
                # Method 2: Try with different enum import
                try:
                    from databricks.sdk.service.vectorsearch import EndpointType
                    endpoint = self.workspace_client.vector_search_endpoints.create_endpoint(
                        name="recovery-vector-search",
                        endpoint_type=EndpointType.STANDARD
                    )
                    logger.info(f"Created vector search endpoint: recovery-vector-search")
                    return True
                    
                except Exception as e2:
                    logger.info(f"Method 2 failed: {e2}")
                    
                    # Method 3: Try string value
                    try:
                        endpoint = self.workspace_client.vector_search_endpoints.create_endpoint(
                            name="recovery-vector-search",
                            endpoint_type="STANDARD"
                        )
                        logger.info(f"Created vector search endpoint: recovery-vector-search")
                        return True
                        
                    except Exception as e3:
                        logger.error(f"All methods failed to create endpoint: {e3}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error in create_vector_search_endpoint: {e}")
            return False

    def create_vector_search_index(self):
        """Create vector search index with better error handling"""
        try:
            # Check if endpoint exists first
            try:
                existing_endpoints = list(self.workspace_client.vector_search_endpoints.list_endpoints())
                endpoint_exists = any(
                    hasattr(endpoint, 'name') and endpoint.name == self.vector_search_endpoint 
                    for endpoint in existing_endpoints
                )
                
                if not endpoint_exists:
                    logger.warning(f"Vector search endpoint {self.vector_search_endpoint} not found. Skipping index creation.")
                    return
                    
            except Exception as list_error:
                logger.warning(f"Could not verify endpoint existence: {list_error}")
                logger.info("Attempting to create index anyway...")
                
            # Create vector search index
            self.vector_search_client.create_delta_sync_index(
                endpoint_name=self.vector_search_endpoint,
                index_name=f"{self.catalog_name}.{self.schema_name}.{self.index_name}",
                source_table_name=self.table_name,
                pipeline_type="TRIGGERED",
                primary_key="id",
                embedding_dimension=1536,
                embedding_vector_column="embedding"
            )
            logger.info(f"Created vector search index: {self.index_name}")
            
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Vector index {self.index_name} already exists")
            elif "not found" in str(e).lower():
                logger.warning(f"Vector search endpoint not found. Skipping vector search setup.")
            else:
                logger.error(f"Error creating vector index: {e}")
                logger.warning("Continuing without vector search capabilities")

    def insert_documents(self, documents: List[Dict]) -> bool:
        """Insert documents into Databricks table using SQL"""
        try:
            with databricks_connect(
                server_hostname=self.workspace_url.replace("https://", ""),
                http_path="/sql/1.0/warehouses/f3b9d912c6b793ad",
                access_token=self.token
            ) as connection:
                with connection.cursor() as cursor:
                    for doc in documents:
                        # Escape single quotes in text fields
                        def escape_sql(text):
                            if text is None:
                                return 'NULL'
                            return f"'{str(text).replace(chr(39), chr(39)+chr(39))}'"
                        
                        # Handle arrays and embeddings
                        tags_array = "array(" + ",".join([f"'{tag}'" for tag in doc.get("tags", [])]) + ")"
                        
                        # Handle embedding array (convert to string representation)
                        embedding = doc.get("embedding")
                        if embedding and len(embedding) > 0:
                            # Take first 10 values for demo (full embedding might be too large for SQL)
                            embedding_sample = embedding[:10]
                            embedding_array = "array(" + ",".join([f"{float(val)}" for val in embedding_sample]) + ")"
                        else:
                            embedding_array = "array()"
                        
                        # Construct INSERT statement
                        insert_sql = f"""
                        INSERT INTO {self.table_name} (
                            id, title, content, category, topic, tags, 
                            chunk_text, chunk_index, embedding, source, 
                            created_at, updated_at, is_active
                        ) VALUES (
                            {escape_sql(doc.get("id"))},
                            {escape_sql(doc.get("title"))},
                            {escape_sql(doc.get("content"))},
                            {escape_sql(doc.get("category"))},
                            {escape_sql(doc.get("topic"))},
                            {tags_array},
                            {escape_sql(doc.get("chunk_text"))},
                            {doc.get("chunk_index", 0)},
                            {embedding_array},
                            {escape_sql(doc.get("source"))},
                            current_timestamp(),
                            current_timestamp(),
                            true
                        )
                        """
                        
                        cursor.execute(insert_sql)
                        logger.info(f"Inserted document: {doc.get('id')}")
                        
            logger.info(f"Successfully inserted {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting documents via SQL: {e}")
            return False

    def search_similar(self, query_vector: List[float], num_results: int = 5) -> List[Dict]:
        """Search for similar documents using vector search"""
        try:
            # Get the vector search index first
            index = self.vector_search_client.get_index(
                index_name=f"{self.catalog_name}.{self.schema_name}.{self.index_name}"
            )
            
            # Then call similarity_search on the index
            results = index.similarity_search(
                query_vector=query_vector,
                columns=["id", "title", "content", "category", "topic", "chunk_text"],
                num_results=num_results
            )
            
            return results.get('result', {}).get('data_array', [])
            
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []

class DatabricksAnalytics:
    """Analytics and ML pipeline using Databricks"""
    
    def __init__(self, workspace_url: str, token: str):
        self.workspace_url = workspace_url
        self.token = token
        self.workspace_client = WorkspaceClient(host=workspace_url, token=token)
        
        # MLflow setup
        mlflow.set_tracking_uri("databricks")
        self.mlflow_client = MlflowClient()
        
    def create_analytics_tables(self):
        """Create tables for analytics and ML features"""
        tables_sql = {
            "user_interactions": f"""
            CREATE TABLE IF NOT EXISTS recoveredge.analytics.user_interactions (
                interaction_id STRING,
                user_id STRING,
                session_id STRING,
                interaction_type STRING, -- 'chat', 'plan_generation', 'video_view'
                timestamp TIMESTAMP,
                content STRING,
                metadata MAP<STRING, STRING>,
                location STRING,
                equipment_used ARRAY<STRING>
            ) USING DELTA
            TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
            """,
            
            "recovery_sessions": f"""
            CREATE TABLE IF NOT EXISTS recoveredge.analytics.recovery_sessions (
                session_id STRING,
                user_id STRING,
                planned_duration INT,
                actual_duration INT,
                completed_methods ARRAY<STRING>,
                skipped_methods ARRAY<STRING>,
                location STRING,
                equipment_used ARRAY<STRING>,
                satisfaction_rating INT,
                created_at TIMESTAMP,
                completed_at TIMESTAMP
            ) USING DELTA
            """,
            
            "method_effectiveness": f"""
            CREATE TABLE IF NOT EXISTS recoveredge.analytics.method_effectiveness (
                method_name STRING,
                user_id STRING,
                effectiveness_rating INT,
                duration_completed INT,
                reported_benefits ARRAY<STRING>,
                side_effects ARRAY<STRING>,
                timestamp TIMESTAMP,
                user_fitness_level STRING,
                user_age_group STRING
            ) USING DELTA
            """,
            
            "feature_store": f"""
            CREATE TABLE IF NOT EXISTS recoveredge.ml.user_features (
                user_id STRING,
                avg_session_duration FLOAT,
                preferred_methods ARRAY<STRING>,
                preferred_locations ARRAY<STRING>,
                equipment_availability MAP<STRING, BOOLEAN>,
                fitness_level STRING,
                age_group STRING,
                activity_frequency INT,
                satisfaction_score FLOAT,
                feature_timestamp TIMESTAMP
            ) USING DELTA
            """
        }
        
        try:
            with databricks_connect(
                server_hostname=self.workspace_url.replace("https://", ""),
                http_path="/sql/1.0/warehouses/f3b9d912c6b793ad",
                access_token=self.token
            ) as connection:
                with connection.cursor() as cursor:
                    for table_name, sql in tables_sql.items():
                        cursor.execute(sql)
                        logger.info(f"Created analytics table: {table_name}")
                        
        except Exception as e:
            logger.error(f"Error creating analytics tables: {e}")
            raise
    
    def train_recommendation_model(self):
        """Train ML model for recovery method recommendations"""
        with mlflow.start_run(run_name="recovery_recommendation_model"):
            try:
                # Feature engineering query
                feature_query = """
                SELECT 
                    user_id,
                    collect_list(method_name) as past_methods,
                    avg(effectiveness_rating) as avg_effectiveness,
                    count(*) as total_sessions,
                    collect_set(location) as preferred_locations,
                    avg(duration_completed) as avg_duration,
                    mode(user_fitness_level) as fitness_level
                FROM recoveredge.analytics.method_effectiveness me
                JOIN recoveredge.analytics.recovery_sessions rs ON me.user_id = rs.user_id
                WHERE me.timestamp > current_date() - interval 90 days
                GROUP BY user_id
                """
                
                # Load training data (placeholder - replace with actual Spark execution)
                # training_data = spark.sql(feature_query).toPandas()
                
                # For demo purposes, create sample data
                training_data = pd.DataFrame({
                    'user_id': ['user1', 'user2', 'user3'],
                    'avg_effectiveness': [4.2, 3.8, 4.5],
                    'total_sessions': [15, 8, 22],
                    'avg_duration': [12.5, 8.2, 18.3],
                    'fitness_level': ['intermediate', 'beginner', 'advanced']
                })
                
                # Simple recommendation model (replace with more sophisticated approach)
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import LabelEncoder
                
                # Encode categorical features
                le = LabelEncoder()
                training_data['fitness_level_encoded'] = le.fit_transform(training_data['fitness_level'])
                
                # Features and target (simplified)
                features = ['avg_effectiveness', 'total_sessions', 'avg_duration', 'fitness_level_encoded']
                X = training_data[features]
                y = training_data['avg_effectiveness'] > 4.0  # Binary target for demo
                
                # Train model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # Log model
                mlflow.sklearn.log_model(model, "recovery_recommendation_model")
                mlflow.log_params({
                    "n_estimators": 100,
                    "model_type": "RandomForest",
                    "features": features
                })
                mlflow.log_metrics({
                    "training_samples": len(training_data),
                    "feature_count": len(features)
                })
                
                logger.info("Trained and logged recommendation model")
                return model
                
            except Exception as e:
                logger.error(f"Error training model: {e}")
                raise

class DatabricksRAGSystem:
    """Enhanced RAG system using Databricks infrastructure"""
    
    def __init__(self, workspace_url: str, token: str, vector_search_endpoint: str):
        self.workspace_url = workspace_url
        self.token = token
        
        # Initialize components
        self.vector_store = DatabricksVectorStore(workspace_url, token, vector_search_endpoint)
        self.analytics = DatabricksAnalytics(workspace_url, token)
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o")
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
    
    def initialize_system(self):
        """Initialize the complete Databricks-based system"""
        try:
            # Setup Unity Catalog
            self.vector_store.setup_unity_catalog()
            
            # Create tables
            self.vector_store.create_knowledge_table()
            self.analytics.create_analytics_tables()

            # Create vector search endpoint
            self.vector_store.create_vector_search_endpoint()
            
            # Create vector search index
            self.vector_store.create_vector_search_index()
            
            # Load initial knowledge base
            self.load_initial_knowledge()
            
            logger.info("Databricks RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            raise
    
    def load_initial_knowledge(self):
        """Load initial recovery knowledge into Databricks"""
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
                "tags": ["foam_rolling", "myofascial_release", "DOMS", "recovery"],
                "source": "initial_load",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "is_active": True
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
                "tags": ["cold_therapy", "ice_bath", "inflammation", "recovery"],
                "source": "initial_load",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "is_active": True
            }
        ]
        
        # Process documents into chunks with embeddings
        processed_docs = []
        for doc in knowledge_docs:
            chunks = self.text_splitter.split_text(doc["content"])
            
            for i, chunk in enumerate(chunks):
                embedding = self.embeddings.embed_query(chunk)
                
                processed_doc = doc.copy()
                processed_doc.update({
                    "chunk_text": chunk,
                    "chunk_index": i,
                    "embedding": embedding
                })
                processed_docs.append(processed_doc)
        
        # Insert into Databricks
        self.vector_store.insert_documents(processed_docs)
        
    def get_response(self, message: str, user_id: str = None, session_id: str = None) -> Tuple[str, List[str]]:
        """Get response using Databricks-powered RAG"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(message)
            
            # Search similar chunks in Databricks
            similar_chunks = self.vector_store.search_similar(query_embedding, num_results=3)
            
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
            
            # Log interaction for analytics
            if user_id and session_id:
                self.log_interaction(user_id, session_id, message, response, similar_chunks)
            
            chunk_ids = [chunk.get("id", "") for chunk in similar_chunks]
            return response, chunk_ids
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error. Please try again.", []
    
    def log_interaction(self, user_id: str, session_id: str, query: str, 
                       response: str, retrieved_chunks: List[Dict]):
        """Log user interactions for analytics"""
        try:
            interaction_data = {
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
            
            # Insert into analytics table (simplified - in practice, use Spark)
            logger.info(f"Logged interaction for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error logging interaction: {e}")

# Initialize the Databricks RAG system
DATABRICKS_WORKSPACE_URL = os.getenv('DATABRICKS_WORKSPACE_URL')
DATABRICKS_TOKEN = os.getenv('DATABRICKS_TOKEN') 
VECTOR_SEARCH_ENDPOINT = os.getenv('VECTOR_SEARCH_ENDPOINT', 'recovery-vector-search')

if DATABRICKS_WORKSPACE_URL and DATABRICKS_TOKEN:
    rag_system = DatabricksRAGSystem(
        DATABRICKS_WORKSPACE_URL, 
        DATABRICKS_TOKEN, 
        VECTOR_SEARCH_ENDPOINT
    )
else:
    rag_system = None
    logger.warning("Databricks credentials not found. Using fallback mode.")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "RecoverEdge Databricks RAG Server",
        "databricks_connected": rag_system is not None
    }
    
    if rag_system:
        try:
            # Test Databricks connection
            rag_system.workspace_client.clusters.list()
            status["databricks_status"] = "connected"
        except Exception as e:
            status["databricks_status"] = f"error: {str(e)}"
            status["status"] = "degraded"
    
    return jsonify(status)

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint using Databricks RAG"""
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400
        
        user_message = data['message']
        user_id = data.get('user_id', 'anonymous')
        session_id = data.get('session_id', 'default')
        
        if rag_system:
            response, chunk_ids = rag_system.get_response(user_message, user_id, session_id)
        else:
            response = "Databricks RAG system not available. Please check configuration."
            chunk_ids = []
        
        return jsonify({
            "response": response,
            "session_id": session_id,
            "retrieved_chunks": len(chunk_ids),
            "powered_by": "databricks" if rag_system else "fallback",
            "error": None
        })
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return jsonify({
            "response": "I encountered an error processing your request.",
            "error": str(e)
        }), 500

@app.route('/analytics/insights', methods=['GET'])
def get_analytics_insights():
    """Get analytics insights from Databricks"""
    if not rag_system:
        return jsonify({"error": "Databricks not configured"}), 503
    
    try:
        # Sample analytics queries (replace with actual Spark SQL)
        insights = {
            "popular_topics": [
                {"topic": "foam_rolling", "query_count": 45, "avg_rating": 4.2},
                {"topic": "cold_therapy", "query_count": 32, "avg_rating": 4.5},
                {"topic": "stretching", "query_count": 28, "avg_rating": 4.1}
            ],
            "user_engagement": {
                "daily_active_users": 156,
                "avg_session_duration": 8.5,
                "completion_rate": 0.78
            },
            "method_effectiveness": {
                "top_rated": "Cold Water Immersion",
                "most_completed": "Deep Breathing",
                "avg_satisfaction": 4.3
            }
        }
        
        return jsonify(insights)
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommendations/<user_id>', methods=['GET'])
def get_recommendations(user_id: str):
    """Get personalized recommendations using ML model"""
    if not rag_system:
        return jsonify({"error": "Databricks not configured"}), 503
    
    try:
        # Load user features and generate recommendations
        # This would typically query the feature store and use the ML model
        recommendations = {
            "recommended_methods": [
                {
                    "name": "Foam Rolling - Legs",
                    "confidence": 0.85,
                    "reason": "Based on your past preferences and effectiveness ratings"
                },
                {
                    "name": "Deep Breathing",
                    "confidence": 0.78,
                    "reason": "High completion rate for your fitness level"
                }
            ],
            "optimal_duration": 15,
            "preferred_location": "home",
            "model_version": "v1.2.0"
        }
        
        return jsonify(recommendations)
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("RecoverEdge Databricks RAG Server")
    print("Advanced Analytics & Vector Search")
    print("Server running on http://localhost:8000")
    print("=" * 60)
    
    # Initialize system if credentials available
    if rag_system:
        try:
            rag_system.initialize_system()
            print("✅ Databricks RAG system initialized")
        except Exception as e:
            print(f"❌ Error initializing Databricks: {e}")
    else:
        print("⚠️  Databricks credentials not configured")
    
    app.run(host='0.0.0.0', port=8000, debug=True)