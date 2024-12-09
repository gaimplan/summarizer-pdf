import os
import textwrap
import logging
from dotenv import load_dotenv
from openai import OpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain_groq import ChatGroq
import time
from groq import RateLimitError
import google.api_core.exceptions
import json
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SimpleNodeParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI clients
client1 = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
client2 = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
client3 = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def open_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

def save_json_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(content, outfile, indent=2)

def gpt_prompt2(prompt, max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        try:
            response = client1.chat.completions.create(
                model="gemma2:9b-instruct-fp16",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            if attempt < max_retries - 1:
                print(f"Rate limit reached. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Error: {e}")
                return None

def gpt_prompt3(prompt, max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        try:
            response = client2.chat.completions.create(
                model="gemma2:latest",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            if attempt < max_retries - 1:
                print(f"Rate limit reached. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Error: {e}")
                return None

def gpt_prompt1(prompt, max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        try:
            response = client1.chat.completions.create(
                model="gemma2:9b-instruct-fp16",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            if attempt < max_retries - 1:
                print(f"Rate limit reached. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Error: {e}")
                return None

def quality_check(chunk, summary):
    prompt = f"""Compare the following original text and summary.

    On a scale from 0% to 100%, how accurately does the summary represent the original text?

    Original Text:
    \"\"\"{chunk}\"\"\"


    Summary:
    \"\"\"{summary}\"\"\"


    Provide only the percentage number."""


    response = gpt_prompt3(prompt)
    # Remove any non-digit characters and convert to float
    try:
        return float(''.join(filter(str.isdigit, response)))
    except ValueError:
        return 0.0  # Return 0 if the response is invalid

def setup_logger():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(filename='logs/summarizer-debug.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger()

def is_valid_response(response):
    invalid_phrases = [
        "i can't", "i cannot", "i'm unable", "i am unable", "i don't have",
        "as an ai language model", "i apologize", "sorry", "unfortunately",
        "i am just", "i was designed", "i cannot provide", "i don't have browsing capabilities",
        "i don't have access", "i cannot access", "i cannot retrieve"
    ]
    response_lower = response.lower()
    return not any(phrase in response_lower for phrase in invalid_phrases)

def clean_topics_output(output):
    # Remove any lines before the first bullet point
    lines = output.strip().split('\n')
    bullet_points = [line for line in lines if line.strip().startswith('* ')]
    return '\n'.join(bullet_points)

# Main script
logger = setup_logger()

# Load PDF file
pdf_reader = PDFReader()
documents = pdf_reader.load_data(file='./input/ai-transparency-ai-codec-technical-note.pdf')

# Initialize text splitter and create chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=40)
# Create node parser
node_parser = SimpleNodeParser()
nodes = node_parser.get_nodes_from_documents(documents, text_splitter=text_splitter)

# Initialize results storage
results = []

# Load prompt templates
notes_prompt_template = open_file("./prompts/summarizer-notes-prompt.txt")
fallback_prompt_template = open_file("./prompts/summarizer-notes-prompt-fallback.txt")

# Process each chunk
print("\nProcessing PDF document chunks...")
logger.info("Processing PDF document chunks...")

for chunk_idx, node in enumerate(nodes):
    chunk_text = node.text
    print(f"\nProcessing chunk {chunk_idx + 1}/{len(nodes)}...")
    
    # Initialize chunk results
    chunk_result = {
        "chunk_id": chunk_idx,
        "chunk_text": chunk_text,
        "notes": None,
        "topics": None,
        "relevancy_percentage": 0
    }
    
    # Process notes for the chunk
    attempt = 0
    max_attempts = 30
    relevancy_threshold = 90.0
    
    while attempt < max_attempts:
        attempt += 1
        
        # Adjust relevancy threshold after 5 attempts
        if attempt > 5:
            relevancy_threshold = 85.0
        
        prompt_modified = notes_prompt_template.format(chunk=chunk_text)
        fallback_prompt_modified = fallback_prompt_template.format(chunk=chunk_text)
        
        # Use different models for different attempts
        if attempt <= 10:
            notes = gpt_prompt1(prompt_modified)
        elif attempt <= 20:
            notes = gpt_prompt2(prompt_modified)
        else:
            notes = gpt_prompt3(prompt_modified)
            
        if notes is None:
            print(f"Failed to get response for attempt {attempt}. Continuing to next attempt.")
            continue
            
        relevancy_percentage = quality_check(chunk_text, notes)
        print(f"Relevancy Accuracy for Notes (Attempt {attempt}): {relevancy_percentage}%")
        
        if relevancy_percentage >= relevancy_threshold:
            chunk_result["notes"] = notes
            chunk_result["relevancy_percentage"] = relevancy_percentage
            break
    
    # Generate topics for the chunk
    if chunk_result["notes"] is not None:
        keytw_template = open_file("./prompts/summarizer-topics-prompt.txt")
        keytw_modified = keytw_template.replace("<<NOTES>>", chunk_result["notes"])
        
        topic_attempt = 0
        max_topic_attempts = 3
        
        while topic_attempt < max_topic_attempts:
            if topic_attempt == 0:
                keytw2 = gpt_prompt1(keytw_modified)
            elif topic_attempt == 1:
                keytw2 = gpt_prompt2(keytw_modified)
            else:
                keytw2 = gpt_prompt3(keytw_modified)
                
            if keytw2 is not None:
                chunk_result["topics"] = clean_topics_output(keytw2)
                break
                
            topic_attempt += 1
            print(f"Failed to generate topics for chunk {chunk_idx}, attempt {topic_attempt}.")
    
    # Add chunk results to main results list
    results.append(chunk_result)
    
    # Save intermediate results after each chunk
    save_json_file('./output/chunk_summaries.json', results)

print("\nProcessing complete!")
logger.info("Processing complete!")
