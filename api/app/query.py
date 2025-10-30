import os
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# --- Configuration ---
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print(GROQ_API_KEY)

# --- Initialize Clients and Models ---

# Initialize a persistent ChromaDB client
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# Initialize the embedding function/model
embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

# Get the collection from ChromaDB
collection = chroma_client.get_collection(
    name=CHROMA_COLLECTION,
    embedding_function=embedding_function
)

# Initialize the Groq client
groq_client = Groq(api_key=GROQ_API_KEY)


# --- Core Query Function ---

def query_rag_engine(query_text: str) -> dict:
    """
    Performs the RAG pipeline: retrieves context and generates an answer.
    """
    print(f"Received query: '{query_text}'")

    # 1. Retrieve relevant context from ChromaDB
    # The query() method automatically handles embedding the query_text.
    results = collection.query(
        query_texts=[query_text],
        n_results=3  # Retrieve the top 3 most relevant chunks
    )

    retrieved_chunks = results['documents'][0]
    if not retrieved_chunks:
        print("No relevant context found in the database.")
        return {
            "answer": "I could not find an answer in the ingested content.",
            "sources": []
        }

    context = "\n---\n".join(retrieved_chunks)
    print(f"Retrieved context:\n{context}")

    # 2. Build the prompt for the LLM
    prompt = f"""
    You are a helpful assistant. Answer the user's question based ONLY on the following context.
    If the answer is not found in the context, say "I could not find an answer in the ingested content."
    Do not use any prior knowledge.

    Context:
    ---
    {context}
    ---

    Question: {query_text}

    Answer:
    """

    # 3. Generate a grounded answer using Groq
    print("Generating answer with Groq...")
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant", # Or another fast model like mixtral-8x7b-32768
        )
        answer = chat_completion.choices[0].message.content
        print(f"Generated answer: {answer}")

        # Extract unique source URLs from the metadata
        source_urls = list(set(meta['source_url'] for meta in results['metadatas'][0]))
        
        return {
            "answer": answer,
            "sources": source_urls
        }
    except Exception as e:
        print(f"Error generating answer with Groq: {e}")
        raise