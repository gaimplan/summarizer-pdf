from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def create_vector_search_index(uri, db_name, collection_name):
    """
    Create a vector search index for the specified collection.
    
    Parameters:
    uri (str): MongoDB connection string
    db_name (str): Name of the database
    collection_name (str): Name of the collection
    """
    try:
        # Connect to MongoDB
        client = MongoClient(uri)
        database = client[db_name]
        collection = database[collection_name]
        
        # Define the index model with correct structure including mappings
        search_index_model = SearchIndexModel(
            definition={
                "mappings": {
                    "dynamic": True,
                    "fields": {
                        "embedding": {
                            "dimensions": 768,
                            "similarity": "cosine",
                            "type": "knnVector"
                        }
                    }
                }
            },
            name="vector_search_index"
        )
        
        # Create the search index
        index_name = collection.create_search_index(model=search_index_model)
        print(f"New search index named '{index_name}' is being created...")
        
        # Poll for index readiness
        print("Polling to check if the index is ready. This may take a few minutes...")
        while True:
            indexes = list(collection.list_search_indexes())
            if indexes and any(idx.get("queryable", False) for idx in indexes):
                break
            time.sleep(5)
            
        print(f"Index '{index_name}' is ready for querying.")
        return index_name
        
    except Exception as e:
        print(f"An error occurred: {str(e)}, full error: {e.args[0]}")
        raise
    
    finally:
        client.close()

def main():
    # Get MongoDB connection string from environment variables
    uri = os.getenv('MONGODB_ATLAS_URI')
    if not uri:
        raise ValueError("MONGODB_ATLAS_URI environment variable not set")
    
    # Database and collection names for your embeddings
    db_name = 'docs'
    collection_name = 'chunk_embeddings'
    
    try:
        index_name = create_vector_search_index(uri, db_name, collection_name)
        print(f"Successfully created vector search index: {index_name}")
    except Exception as e:
        print(f"Failed to create vector search index: {str(e)}, full error: {e.args[0]}")

if __name__ == "__main__":
    main()
