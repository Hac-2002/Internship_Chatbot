from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS
from langchain.text_splitter import CharacterTextSplitter
import logging
from datetime import datetime
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.llms import OpenAI

from .config import Config
from .exceptions import (
    ChatbotException,
    InitializationError,
    DocumentProcessingError,
    QuestionProcessingError,
    APIError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChatbotSystem:
    """Main chatbot system handling document processing and question answering."""
    
    def __init__(self):
        """Initialize the chatbot system with necessary components."""
        try:
            self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
            self._initialize_components()
            self._load_and_process_documents()
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise InitializationError(f"Failed to initialize chatbot: {str(e)}")

    def _initialize_components(self) -> None:
        """Initialize basic components for the chatbot system."""
        self.loader = WebBaseLoader(Config.BASE_URL)
        self.loader.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.loader.session.verify = True
        self.loader.session.timeout = Config.SCRAPE_TIMEOUT
        self.llm = OpenAI(
            api_key=Config.OPENAI_API_KEY,
            temperature=Config.TEMPERATURE
        )
        self.document_embeddings = []
        self.documents = []

    def _load_and_process_documents(self) -> None:
        """Load and process documents with error handling."""
        try:
            documents = self.loader.load()
            text_splitter = CharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            self.documents = text_splitter.split_documents(documents)
            
            # Create embeddings for all documents
            texts = [doc.page_content for doc in self.documents]
            self.document_embeddings = self.model.encode(texts)
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to load or process documents: {str(e)}")

    def _find_relevant_documents(self, query: str) -> List[str]:
        """
        Find most relevant documents using cosine similarity.
        
        Args:
            query: The question to find relevant documents for.
            
        Returns:
            List of relevant document contents.
        """
        query_embedding = self.model.encode([query])[0]
        
        # Calculate similarities
        similarities = np.dot(self.document_embeddings, query_embedding) / (
            np.linalg.norm(self.document_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k most similar documents
        top_k_indices = np.argsort(similarities)[-Config.TOP_K_RESULTS:][::-1]
        return [self.documents[i].page_content for i in top_k_indices]

    def get_answer(self, question: str) -> str:
        """
        Get answer for a question using relevant documents and LLM.
        
        Args:
            question: The question to answer.
            
        Returns:
            Generated answer based on relevant documents.
            
        Raises:
            QuestionProcessingError: If processing the question fails.
        """
        try:
            relevant_docs = self._find_relevant_documents(question)
            context = "\n".join(relevant_docs)
            
            prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Answer:"""
            
            return self.llm.predict(prompt)
            
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            raise QuestionProcessingError(f"Failed to get answer: {str(e)}")

# Initialize Flask app and API
app = Flask(__name__)
CORS(app)
api = Api(app)

# Initialize chatbot
print("Initializing chatbot system... This might take a few minutes...")
try:
    chatbot = ChatbotSystem()
    print("Chatbot system initialized successfully!")
except ChatbotException as e:
    logger.error(f"Failed to initialize chatbot: {str(e)}")
    chatbot = None
    print(f"Failed to initialize chatbot: {str(e)}")

class ChatEndpoint(Resource):
    """REST API endpoint for chat functionality."""
    
    def post(self):
        """
        Handle POST requests for chat.
        
        Returns:
            JSON response containing answer or error message.
        """
        try:
            if not chatbot:
                raise APIError("Chatbot system is not initialized")
            
            data = request.get_json()
            if not data or 'question' not in data:
                raise APIError("Question is required")
            
            question = data['question'].strip()
            if not question:
                raise APIError("Question cannot be empty")
                
            answer = chatbot.get_answer(question)
            
            return {
                "answer": answer,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
        except APIError as e:
            logger.error(f"API error: {str(e)}")
            return {"error": str(e)}, 400
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {"error": "An unexpected error occurred while processing your request"}, 500

# Add endpoint
api.add_resource(ChatEndpoint, '/chat')

if __name__ == '__main__':
    banner = """
    =================================
    Course Chatbot API
    =================================
    
    Server is starting...
    - Endpoint: http://localhost:{}/chat
    - Method: POST
    - Body format: {{"question": "your question here"}}
    
    =================================
    """.format(Config.API_PORT)
    
    print(banner)
    
    app.run(
        host=Config.API_HOST,
        port=Config.API_PORT,
        debug=Config.DEBUG_MODE
    )
