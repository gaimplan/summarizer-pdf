import os
import json
from dotenv import load_dotenv
from pymongo import MongoClient
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.core import StorageContext
from llama_index.core.embeddings import BaseEmbedding
import requests
from typing import List
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables from .env file
load_dotenv()

class OllamaEmbedding(BaseEmbedding):
    def __init__(self):
        super().__init__()
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _get_single_embedding(self, text: str) -> List[float]:
        """Get embedding using Ollama's nomic-embed-text model with retry logic."""
        try:
            response = requests.post('http://localhost:11434/api/embeddings',
                                  json={'model': 'nomic-embed-text', 'prompt': text})
            return response.json()['embedding']
        except Exception as e:
            print(f"Error getting embedding, retrying... Error: {str(e)}")
            time.sleep(1)
            raise

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_single_embedding(text)
        
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_single_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

def create_document_context(chunk: dict) -> str:
    """Create rich context from document fields."""
    context_parts = []
    
    if chunk.get('notes'):
        # Remove HTML-like tags and clean up the notes
        notes = chunk['notes'].replace('<summary>', '').replace('</summary>', '').strip()
        context_parts.append(f"Summary: {notes}")
    
    if chunk.get('topics'):
        # Clean up topics and join them
        topics = chunk['topics'].replace('* ', '').replace('\n', ', ').strip()
        context_parts.append(f"Topics: {topics}")
    
    return ' | '.join(context_parts)

def main():
    # MongoDB connection details
    uri = os.environ.get('MONGODB_ATLAS_URI')
    if uri is None:
        raise ValueError("MONGODB_ATLAS_URI environment variable is not set")

    # Set the MONGODB_URI environment variable that llama_index expects
    os.environ["MONGODB_URI"] = uri

    db_name = "docs"
    collection_name = "chunk_embeddings"

    # Initialize MongoDB client and create vector store
    client = MongoClient(uri, tls=True, tlsAllowInvalidCertificates=True)
    vector_store = MongoDBAtlasVectorSearch(
        client=client,
        db_name=db_name,
        collection_name=collection_name,
        index_name="default",
        embedding_dimension=768,  # Google's text-embedding-004 dimension
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OllamaEmbedding()

    # Load and process the JSON file
    with open('../output/chunk_summaries.json', 'r') as f:
        chunks = json.load(f)

    # Process each chunk
    for chunk in chunks:
        print(f"Processing chunk {chunk['chunk_id']}")
        
        # Create document context
        doc_context = create_document_context(chunk)
        
        # Combine context and chunk text for embedding
        text_to_embed = f"{doc_context} | Content: {chunk['chunk_text']}"
        
        # Get embedding
        embedding = embed_model._get_text_embedding(text_to_embed)
        
        # Create document for MongoDB
        doc = {
            "chunk_id": chunk['chunk_id'],
            "text": chunk['chunk_text'],  # Original chunk without context
            "text_with_context": text_to_embed,  # Full text that was embedded
            "embedding": embedding,
            "doc_context": doc_context,
            "topics": chunk.get('topics', ''),
            "notes": chunk.get('notes', ''),
            "relevancy_percentage": chunk.get('relevancy_percentage', 0),
            "creation_date": datetime.now().isoformat()
        }
        
        # Insert into MongoDB using the correct method
        collection = client[db_name][collection_name]
        collection.insert_one(doc)
        
        print(f"Processed and embedded chunk {chunk['chunk_id']}")

    print("Processing complete. All chunks have been embedded and stored in MongoDB Atlas.")
    print("Please ensure you have created the appropriate Atlas Search index in your MongoDB Atlas cluster.")
    
    client.close()

if __name__ == "__main__":
    main() 