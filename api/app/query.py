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
groq_enrich_client = Groq(api_key=GROQ_API_KEY)

# --- Core Query Function ---

def query_rag_engine(query_text: str) -> dict:
    """
    Performs the RAG pipeline:
    1. Improves/rephrases user query using Groq LLM with mixtral-8x7b-32768.
    2. Retrieves context from ChromaDB using enriched query.
    3. Uses context to answer the question (RAG step) via Groq LLM with llama-3.1-8b-instant.
    """
    print(f"Received raw query: '{query_text}'")

    # Step 1: Enrich/rephrase the user query using Groq LLM (using mixtral model)
    enrich_prompt = (
        "Please rephrase or enrich the following question to make it clearer and more specific for a customer assistant. "
        "Ensure the question is well-formed, focused, and easy for an AI to understand:\n\n"
        f"Original Question: {query_text}\n\n"
        "Improved Question:"
    )

    try:
        enrich_completion = groq_enrich_client.chat.completions.create(
            messages=[{"role": "user", "content": enrich_prompt}],
            model="mixtral-8x7b-32768",  # <--- Different model for enrichment
        )
        improved_query_text = enrich_completion.choices[0].message.content.strip()
        print(f"Enriched query: '{improved_query_text}'")
    except Exception as e:
        print(f"Error enriching query with Groq: {e}")
        # Fallback: use the raw query (better than nothing)
        improved_query_text = query_text

    # Step 2: Retrieve relevant context from ChromaDB using enriched query
    results = collection.query(
        query_texts=[improved_query_text],
        n_results=3  # Retrieve the top 3 most relevant chunks
    )

    retrieved_chunks = results['documents'][0]
    if not retrieved_chunks:
        print("No relevant context found in the database.")
        return {
            "answer": "I could not find an answer in the ingested content.",
            "sources": [],
            "enriched_query": improved_query_text  # Optional: show what query was used
        }

    context = "\n---\n".join(retrieved_chunks)
    print(f"Retrieved context:\n{context}")

    # Step 3: Build the prompt for the LLM (use the improved query, answer with llama model)
    prompt = f"""
    You are a helpful assistant. Answer the user's question based ONLY on the following context.
    If the answer is not found in the context, say "I could not find an answer in the ingested content."
    Do not use any prior knowledge.

    Context:
    ---
    {context}
    ---

    Question: {improved_query_text}

    Answer:
    """

    print("Generating answer with Groq...")
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",  # <--- your existing fast model for answers
        )
        answer = chat_completion.choices[0].message.content
        print(f"Generated answer: {answer}")

        # Extract unique source URLs from the metadata
        source_urls = list(set(meta.get('source_url', '') for meta in results['metadatas'][0]))

        return {
            "answer": answer,
            "sources": source_urls,
            "enriched_query": improved_query_text
        }
    except Exception as e:
        print(f"Error generating answer with Groq: {e}")
        raise 