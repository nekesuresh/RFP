import chromadb
from sentence_transformers import SentenceTransformer

# New persistent client path
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection("rag_collection")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def add_to_vector_db(texts: list[str], ids: list[str], metadatas: list[dict]):
    embeddings = embedder.encode(texts).tolist()
    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)

def query_vector_db(query: str, n_results: int = 3):
    try:
        embedding = embedder.encode([query]).tolist()[0]
        results = collection.query(query_embeddings=[embedding], n_results=n_results)
        docs = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results.get('metadatas') and results['metadatas'] else [{} for _ in docs]
        # Remove duplicates by text while preserving order
        seen = set()
        unique_chunks = []
        for doc, meta in zip(docs, metadatas):
            if doc and doc not in seen:
                chunk_info = {"text": doc, "page": meta.get("page", None), "para": meta.get("para", None)}
                unique_chunks.append(chunk_info)
                seen.add(doc)
        # Defensive: always return a list of dicts with 'text' key
        if not unique_chunks:
            return []
        filtered_chunks = [chunk for chunk in unique_chunks if isinstance(chunk, dict) and 'text' in chunk]
        return filtered_chunks
    except Exception as e:
        import logging
        logging.error(f"Error in query_vector_db: {e}")
        return []

def get_all_paragraph_chunks():
    """Fetch all paragraph chunks from the vector DB."""
    try:
        results = collection.get()
        if not results:
            return []
        docs = results['documents'] if results.get('documents') else []
        metadatas = results['metadatas'] if results.get('metadatas') else [{} for _ in docs]
        chunks = []
        for doc, meta in zip(docs, metadatas):
            if doc:
                chunk_info = {"text": doc, "page": meta.get("page", None), "para": meta.get("para", None)}
                chunks.append(chunk_info)
        return chunks
    except Exception as e:
        import logging
        logging.error(f"Error in get_all_paragraph_chunks: {e}")
        return []