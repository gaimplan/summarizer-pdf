import os
import requests
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
from transformers import AutoTokenizer
from typing import List, Dict

# Create logs directory if it doesn't exist
os.makedirs('../logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    filename='../logs/vector_search.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add console logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Reduce MongoDB logging noise
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Load environment variables and initialize FastAPI
load_dotenv()
app = FastAPI()

# MongoDB connection setup
class MongoDBConnection:
    def __init__(self):
        self.uri = os.environ.get('MONGODB_ATLAS_URI')
        self.client = MongoClient(self.uri, tls=True, tlsAllowInvalidCertificates=True)
        self.db = self.client['docs']
        self.embeddings_collection = self.db['chunk_embeddings']
        self.customers_collection = self.db['customers']
        
        # Verify connection and log collection info
        self.verify_connection()
    
    def verify_connection(self):
        try:
            doc_count = self.embeddings_collection.count_documents({})
            sample_doc = self.embeddings_collection.find_one({})
            logging.info(f"Connected to MongoDB. Embeddings collection count: {doc_count}")
            logging.info(f"Sample document structure: {list(sample_doc.keys()) if sample_doc else 'No documents found'}")
        except Exception as e:
            logging.error(f"MongoDB connection error: {str(e)}")
            raise

# Initialize MongoDB connection
db = MongoDBConnection()

# Initialize tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
    logging.info("Tokenizer initialized successfully")
except Exception as e:
    logging.error(f"Tokenizer initialization failed: {str(e)}")
    raise

class EmbeddingService:
    @staticmethod
    def get_embedding(text: str) -> List[float]:
        """Get embedding vector from Ollama service."""
        logging.info(f"Generating embedding for text length: {len(text)}")
        try:
            response = requests.post(
                'http://localhost:11434/api/embeddings',
                json={'model': 'nomic-embed-text', 'prompt': text}
            )
            response.raise_for_status()
            embedding = response.json()['embedding']
            logging.info(f"Embedding generated successfully. Vector length: {len(embedding)}")
            return embedding
        except requests.RequestException as e:
            logging.error(f"Embedding generation failed: {str(e)}")
            raise

class Query(BaseModel):
    query_text: str
    top_k: int = 20
    min_score: float = 0.0

class SearchService:
    @staticmethod
    def process_search_results(results: List[Dict], customers_collection) -> List[Dict]:
        """Process search results with document context."""
        processed_results = []
        for result in results:
            result_info = {
                "chunk_id": result.get('chunk_id', 'N/A'),
                "text": result.get('text', 'N/A'),
                "doc_context": result.get('doc_context', 'N/A'),
                "topics": result.get('topics', 'N/A'),
                "score": result.get('score', 0.0),
                "relevancy_percentage": result.get('relevancy_percentage', 0)
            }
            processed_results.append(result_info)
        return processed_results

@app.post('/search')
async def search(query: Query):
    logging.info(f"Processing search query: {query.query_text}")
    
    if not query.query_text.strip():
        raise HTTPException(status_code=400, detail='query_text cannot be empty')
    
    try:
        # Get embedding for query
        query_embedding = EmbeddingService.get_embedding(query.query_text)
        
        # Enhanced MongoDB aggregation pipeline using vectorSearch
        pipeline = [
            {
                '$vectorSearch': {
                    'index': 'vector_search_index',
                    'path': 'embedding',
                    'queryVector': query_embedding,
                    'numCandidates': 150,
                    'limit': query.top_k
                }
            },
            {
                '$addFields': {
                    'score': {
                        '$meta': 'vectorSearchScore'
                    }
                }
            },
            {
                '$match': {
                    'score': {'$gte': query.min_score}
                }
            }
        ]
        
        # Execute search
        results = list(db.embeddings_collection.aggregate(pipeline))
        logging.info(f"Found {len(results)} matching documents")
        
        # Process results
        processed_results = SearchService.process_search_results(results, db.customers_collection)
        
        # Calculate token counts
        total_tokens = 0
        for result in processed_results:
            text_elements = [
                result['text'],
                result.get('doc_context', '')
            ]
            total_tokens += sum(len(tokenizer.encode(str(text), add_special_tokens=False))
                              for text in text_elements if text)
        
        # Prepare response
        response_data = {
            'results': processed_results,
            'total_tokens': total_tokens,
            'query': query.query_text
        }
        
        logging.info(f"Search completed. Total tokens: {total_tokens}")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5001)
