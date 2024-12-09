# PDF Document Summarizer & Vector Search System

A summarizer system that processes PDF documents into searchable chunks using RAG (Retrieval Augmented Generation) for summarization, Ollama for embedding generation, and MongoDB Atlas for vector storage and search. The system enables efficient semantic search across processed document chunks with their associated summaries, notes, and topics.

## Prerequisites
- Python 3.8+
- MongoDB Atlas account
- Ollama installed locally

## Setup Environment
1. Install dependencies:
```
pip install -r requirements.txt
```

2. Set up environment variables:
```
cp .example.env .env
```
3. Create the MongoDB Atlas Vector Database

## Core Components

### Document Processing Pipeline
- `rag_summarizer.py`: Initial document processor
  - Processes PDF documents from input folder
  - Chunks documents into manageable segments
  - Generates summaries, notes, and topics for each chunk
  - Creates `chunk_summaries.json` for embedding generation

### Data Processing & Embedding Pipeline
- `upload-embed-chunks.py`: Processes chunks and generates embeddings
  - Reads from `chunk_summaries.json`
  - Generates embeddings via Ollama
  - Uploads enriched chunks and embeddings to MongoDB Atlas
- `create-index-search.py`: Manages vector search indexes
  - Creates and configures indexes in MongoDB Atlas
  - Enables efficient similarity search
- `output/chunk_summaries.json`: Enriched chunk storage
  - Contains chunk text
  - Includes generated summaries
  - Stores topic classifications
  - Used as source for embedding generation

### Search Server
- `vector-search-ollama-5001.py`: Main search service
  - Runs on port 5001
  - Handles vector search requests
  - Integrates with Ollama for embedding generation
  - Performs similarity search in MongoDB Atlas

### Testing Components
- `test-vector-search.py`: Validation suite
  - Tests search functionality
  - Verifies system integration

## Workflow Sequence

1. **Document Processing & Summarization**
   - Process PDF documents from input folder
   - Generate chunk summaries, notes, and topics
   - Create enriched chunk_summaries.json
   ```
   python rag_summarizer.py
   ```

2. **Data Processing & Embedding**
   - Read from chunk_summaries.json
   - Generate embeddings using Ollama
   - Store enriched chunks and embeddings in MongoDB Atlas
   ```
   python embeddings/upload-embed-chunks.py
   ```

3. **Index Creation**
   - Create vector search indexes in MongoDB Atlas
   ```
   python embeddings/create-index-search.py
   ```

4. **Start Search Server**
   - Launch the vector search server on port 5001
   - Handles incoming search requests
   - Generates embeddings for queries
   - Performs similarity search
   ```
   python embeddings/vector-search-ollama-5001.py
   ```

5. **Test Search Functionality**
   - Run test queries against the search server
   ```
   python embeddings/test-vector-search.py
   ```

