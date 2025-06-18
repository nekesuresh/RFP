import chromadb
from sentence_transformers import SentenceTransformer

# New persistent client path
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection("rag_collection")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def add_to_vector_db(texts: list[str], ids: list[str]):
    embeddings = embedder.encode(texts).tolist()
    collection.add(documents=texts, embeddings=embeddings, ids=ids)

def query_vector_db(query: str, n_results: int = 3):
    embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[embedding], n_results=n_results)
    return results['documents'][0] if results['documents'] else []