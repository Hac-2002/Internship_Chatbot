import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the chatbot application."""
    
    # API Settings
    API_PORT = int(os.getenv('API_PORT', 5000))
    API_HOST = os.getenv('API_HOST', '127.0.0.1')
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'True').lower() == 'true'
    
    # OpenAI Settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found in environment variables")
    
    # Document Processing Settings
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
    
    # Web Scraping Settings
    BASE_URL = os.getenv('BASE_URL', 'https://brainlox.com/courses/category/technical')
    SCRAPE_TIMEOUT = int(os.getenv('SCRAPE_TIMEOUT', 30))
    
    # Search Settings
    TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', 3))
    
    # Model Settings
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
