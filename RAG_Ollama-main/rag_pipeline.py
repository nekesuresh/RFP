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
    embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[embedding], n_results=n_results)
    docs = results['documents'][0] if results['documents'] else []
    metadatas = results['metadatas'][0] if results.get('metadatas') and results['metadatas'] else [{} for _ in docs]
    # Remove duplicates by text while preserving order
    seen = set()
    unique_chunks = []
    for doc, meta in zip(docs, metadatas):
        if doc not in seen:
            chunk_info = {"text": doc, "page": meta.get("page", None), "para": meta.get("para", None)}
            unique_chunks.append(chunk_info)
            seen.add(doc)
    return unique_chunks