"""
Advanced Python Server with RAG and LangChain for RecoverEdge
This server uses RAG (Retrieval Augmented Generation) and LangChain
Run with: python rag_server.py
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
from typing import List, Dict, Any

# LangChain imports
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# For using local models (optional)
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.llms import LlamaCpp

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# You'll need to set your OpenAI API key as an environment variable
# export OPENAI_API_KEY='your-api-key-here'

class RecoveryRAGSystem:
    def __init__(self):
        """Initialize the RAG system with recovery knowledge base"""
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
        self.vector_store = None
        self.qa_chain = None
        self.conversation_memory = {}
        
        # Initialize the knowledge base
        self._setup_knowledge_base()
        
    def _setup_knowledge_base(self):
        """Create the knowledge base with recovery-specific content"""
        # Recovery knowledge documents
        recovery_docs = [
            Document(
                page_content="""
                Foam Rolling Techniques and Benefits:
                Foam rolling is a form of self-myofascial release that helps break up fascial adhesions and scar tissue.
                Benefits include: increased blood flow, reduced muscle soreness, improved range of motion, and faster recovery.
                Technique: Apply moderate pressure and roll slowly (1 inch per second) over muscle groups.
                Duration: 30-60 seconds per muscle group, focusing on tight areas.
                Common areas: IT band, quadriceps, hamstrings, calves, back, and glutes.
                Avoid rolling directly on joints, bones, or the lower back.
                Best performed when muscles are warm, either after light activity or a warm shower.
                Research shows foam rolling can reduce DOMS by up to 30% when used consistently.
                """,
                metadata={"topic": "foam_rolling", "category": "self_massage"}
            ),
            
            Document(
                page_content="""
                Cold Therapy and Cryotherapy:
                Cold therapy reduces inflammation and speeds recovery through vasoconstriction.
                Ice baths: 50-60°F (10-15°C) for 10-15 minutes post-workout.
                Cold showers: 2-3 minutes of cold water after regular shower.
                Contrast therapy: Alternate between hot (100-104°F) and cold (50-60°F) water.
                Benefits: Reduced inflammation, decreased muscle damage, improved recovery time.
                Timing: Most effective within 1 hour post-exercise.
                Frequency: 3-4 times per week for athletes, less for recreational exercisers.
                Precautions: Never apply ice directly to skin, limit sessions to prevent hypothermia.
                Research indicates 11-15 minutes of cold exposure weekly optimizes benefits.
                """,
                metadata={"topic": "cold_therapy", "category": "temperature"}
            ),
            
            Document(
                page_content="""
                Stretching and Flexibility:
                Static stretching: Hold positions for 20-30 seconds, 2-4 sets per muscle group.
                Dynamic stretching: Best for warm-ups, involves controlled movements.
                PNF stretching: Contract-relax method for advanced flexibility gains.
                Post-workout stretching reduces muscle tension and improves flexibility.
                Key muscle groups: hamstrings, hip flexors, quadriceps, calves, shoulders, back.
                Breathing: Deep, controlled breathing enhances stretch effectiveness.
                Frequency: Daily stretching ideal, minimum 3-4 times per week.
                Avoid bouncing or forcing stretches beyond comfortable range.
                Research shows regular stretching can improve performance and reduce injury risk by 15-20%.
                """,
                metadata={"topic": "stretching", "category": "flexibility"}
            ),
            
            Document(
                page_content="""
                Sleep and Recovery:
                Sleep is crucial for muscle repair and growth hormone release.
                Optimal duration: 7-9 hours per night for adults, 8-10 for athletes.
                Sleep stages: Deep sleep (stages 3-4) most important for physical recovery.
                Sleep hygiene: Cool room (65-68°F), dark environment, consistent schedule.
                Pre-sleep routine: Avoid screens 1 hour before bed, light stretching, meditation.
                Nutrition: Avoid large meals 3 hours before sleep, limit caffeine after 2 PM.
                Recovery hormones: Growth hormone peaks during deep sleep.
                Sleep debt: Accumulated lack of sleep impairs recovery and performance.
                Naps: 20-30 minute power naps can boost recovery without affecting night sleep.
                Research shows poor sleep can reduce muscle protein synthesis by up to 18%.
                """,
                metadata={"topic": "sleep", "category": "recovery"}
            ),
            
            Document(
                page_content="""
                Nutrition for Recovery:
                Post-workout window: 30-60 minutes optimal for nutrient uptake.
                Protein requirements: 20-30g within 2 hours post-exercise.
                Carbohydrates: 0.5-1g per kg body weight to replenish glycogen.
                Hydration: Replace 150% of fluid lost during exercise.
                Electrolytes: Sodium, potassium, magnesium crucial for recovery.
                Anti-inflammatory foods: Berries, fatty fish, leafy greens, nuts.
                Supplements: Creatine, BCAAs, omega-3s may aid recovery.
                Meal timing: Regular meals every 3-4 hours supports recovery.
                Avoid: Excessive alcohol, processed foods, excessive sugar.
                Research shows proper nutrition can reduce recovery time by 30-50%.
                """,
                metadata={"topic": "nutrition", "category": "diet"}
            ),
            
            Document(
                page_content="""
                Breathing Techniques for Recovery:
                Box breathing: 4-4-4-4 count (inhale-hold-exhale-hold).
                Diaphragmatic breathing: Deep belly breathing activates parasympathetic system.
                Wim Hof method: Combines breathing with cold exposure for recovery.
                4-7-8 breathing: Inhale 4, hold 7, exhale 8 - promotes relaxation.
                Benefits: Reduced cortisol, improved HRV, faster recovery.
                Timing: 5-10 minutes post-workout or before sleep.
                Frequency: Daily practice enhances benefits.
                Integration: Combine with stretching or meditation.
                Research shows controlled breathing can reduce stress hormones by 23%.
                """,
                metadata={"topic": "breathing", "category": "relaxation"}
            ),
            
            Document(
                page_content="""
                Active Recovery Methods:
                Light cardio: 20-30 minutes at 50-60% max heart rate.
                Swimming: Low-impact full-body recovery exercise.
                Yoga: Combines stretching, breathing, and mindfulness.
                Walking: Simple and effective for promoting blood flow.
                Cycling: Easy spinning helps flush metabolic waste.
                Benefits: Increased blood flow, reduced stiffness, mental recovery.
                Frequency: 1-2 active recovery days per week.
                Intensity: Should feel easy, conversational pace.
                Duration: 20-45 minutes typically sufficient.
                Research indicates active recovery can reduce next-day soreness by 40%.
                """,
                metadata={"topic": "active_recovery", "category": "movement"}
            ),
            
            Document(
                page_content="""
                Massage and Soft Tissue Work:
                Massage gun: 1-2 minutes per muscle group, avoid bones and joints.
                Sports massage: Professional treatment 1-2 times monthly.
                Self-massage: Tennis balls, lacrosse balls for trigger points.
                Techniques: Effleurage (stroking), petrissage (kneading), friction.
                Benefits: Improved circulation, reduced adhesions, relaxation.
                Timing: 2-24 hours post-exercise optimal.
                Pressure: Moderate pressure most effective, pain indicates too much.
                Areas: Focus on worked muscles and common tight spots.
                Contraindications: Acute injury, inflammation, skin conditions.
                Studies show massage can reduce DOMS markers by up to 30%.
                """,
                metadata={"topic": "massage", "category": "soft_tissue"}
            )
        ]
        
        # Create vector store from documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        splits = []
        for doc in recovery_docs:
            splits.extend(text_splitter.split_documents([doc]))
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./recovery_knowledge_base"
        )
        
        # Create QA chain with custom prompt
        prompt_template = """You are a knowledgeable recovery and fitness assistant specializing in post-workout recovery techniques.
        Use the following context to answer the question. If you don't know the answer based on the context, say so.
        Always provide evidence-based advice and remind users to consult healthcare professionals for medical concerns.
        
        Context: {context}
        
        Question: {question}
        
        Helpful Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 relevant chunks
            ),
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def get_response(self, message: str, conversation_id: str = "default") -> str:
        """Get response using RAG system"""
        try:
            # Use the QA chain to get response
            response = self.qa_chain.run(message)
            return response
        except Exception as e:
            logger.error(f"Error in RAG system: {str(e)}")
            return "I apologize, but I encountered an error processing your question. Please try again."

# Initialize the RAG system
rag_system = RecoveryRAGSystem()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "RecoverEdge RAG Server",
        "vector_store_status": "active" if rag_system.vector_store else "not initialized"
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint using RAG system"""
    try:
        data = request.json
        logger.info(f"Received request: {data}")
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "No message provided"
            }), 400
        
        user_message = data['message']
        conversation_history = data.get('conversationHistory', [])
        
        # Get response from RAG system
        response = rag_system.get_response(user_message)
        
        logger.info(f"User: {user_message}")
        logger.info(f"Assistant: {response[:100]}...")  # Log first 100 chars
        
        return jsonify({
            "response": response,
            "error": None
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            "response": "",
            "error": f"Server error: {str(e)}"
        }), 500

@app.route('/add_knowledge', methods=['POST'])
def add_knowledge():
    """Endpoint to add new knowledge to the system"""
    try:
        data = request.json
        if not data or 'content' not in data:
            return jsonify({"error": "No content provided"}), 400
        
        # Create new document
        new_doc = Document(
            page_content=data['content'],
            metadata=data.get('metadata', {})
        )
        
        # Add to vector store
        rag_system.vector_store.add_documents([new_doc])
        
        return jsonify({
            "status": "success",
            "message": "Knowledge added successfully"
        })
        
    except Exception as e:
        logger.error(f"Error adding knowledge: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("RecoverEdge RAG/LangChain Server")
    print("Server running on http://localhost:8000")
    print("Make sure OPENAI_API_KEY is set in environment")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("WARNING: OPENAI_API_KEY not found in environment variables!")
        print("Set it using: export OPENAI_API_KEY='your-key-here'")
    
    app.run(host='0.0.0.0', port=8000, debug=True)
