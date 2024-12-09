import requests
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

def check_server_status():
    """Check if the FastAPI server is running"""
    try:
        response = requests.get('http://localhost:5001/')
        return True
    except requests.ConnectionError:
        return False

def search_similar_chunks(query_text, top_k=10, min_score=0.0):
    """
    Search for similar chunks using the vector search API
    
    Parameters:
    query_text (str): Text to search for
    top_k (int): Number of results to return
    min_score (float): Minimum similarity score threshold
    """
    try:
        # Check if server is running
        if not check_server_status():
            print("Error: Vector search server is not running!")
            print("Please start the server by running: python vector-search-ollama-5001.py")
            return

        # Call the FastAPI endpoint
        response = requests.post(
            'http://localhost:5001/search',
            json={
                'query_text': query_text,
                'top_k': top_k,
                'min_score': min_score
            }
        )
        
        # Handle error responses
        if response.status_code != 200:
            error_detail = response.json().get('detail', 'No error details provided')
            print(f"Server Error (Status {response.status_code}): {error_detail}")
            return
        
        # Get results
        results = response.json()
        
        # Check if we got any results
        if not results.get('results'):
            print("\nNo results found for your query.")
            return
        
        # Print results
        print("\nSearch Results for:", query_text)
        print("-" * 80)
        
        for doc in results['results']:
            print(f"\nChunk ID: {doc.get('chunk_id')}")
            print(f"Context: {doc.get('doc_context')}")
            print(f"Topics: {doc.get('topics')}")
            print(f"Content: {doc.get('text')}")
            print(f"Score: {doc.get('score'):.4f}")
            print("-" * 80)
            
        print(f"\nTotal tokens: {results['total_tokens']}")
            
    except requests.RequestException as e:
        print(f"API request failed: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Server response: {e.response.text}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

def main():
    while True:
        # Prompt user for search query
        query = input("\nEnter your search query (or 'quit' to exit): ").strip()
        
        if query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            sys.exit(0)
            
        if not query:
            print("Please enter a valid search query.")
            continue
        
        try:
            search_similar_chunks(query)
        except Exception as e:
            print(f"Search failed: {str(e)}")

if __name__ == "__main__":
    main()
