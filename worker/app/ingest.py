import os
import httpx
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# --- Configuration ---
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")

# --- ChromaDB Client ---
# Initialize a persistent ChromaDB client that connects to our server.
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# --- Embedding Function ---
# This function will be used by ChromaDB to automatically create embeddings.
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

# --- Get or Create Collection ---
# This ensures our collection exists in ChromaDB.
collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    embedding_function=embedding_function
)

# --- Core Ingestion Functions ---

def fetch_and_clean_text(url: str) -> str:
    """Fetches content from a URL and cleans it to get raw text."""
    print(f"Fetching content from {url}...")
    try:
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()  # Will raise an exception for 4xx/5xx responses
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)
        
        print(f"Successfully fetched and cleaned text from {url}.")
        return cleaned_text
    except httpx.HTTPStatusError as e:
        print(f"HTTP error fetching {url}: {e}")
        raise
    except Exception as e:
        print(f"An error occurred during fetch/clean of {url}: {e}")
        raise


def chunk_text(text: str) -> list[str]:
    """Splits a long text into smaller, manageable chunks."""
    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    print(f"Text split into {len(chunks)} chunks.")
    return chunks


def store_chunks_in_db(url: str, chunks: list[str]):
    """Stores text chunks and their metadata in ChromaDB."""
    if not chunks:
        print("No chunks to store.")
        return

    print(f"Storing {len(chunks)} chunks in ChromaDB...")
    # Create unique IDs for each chunk to prevent duplicates
    ids = [f"{url}_{i}" for i, _ in enumerate(chunks)]
    
    # Associate metadata with each chunk
    metadatas = [{"source_url": url} for _ in chunks]
    
    # Add the documents to the collection. ChromaDB will handle the embedding.
    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas
    )
    print("Successfully stored chunks in ChromaDB.") 