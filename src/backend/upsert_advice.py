import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from advice_catalog import ADVICE_CATALOG

def upsert_to_pinecone():
    load_dotenv()
    
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "loan-advice-index")
    
    if not api_key:
        print("ERROR: PINECONE_API_KEY not found in .env")
        return
        
    print(f"Connecting to Pinecone index '{index_name}'...")
    pc = Pinecone(api_key=api_key)
    
    try:
        index = pc.Index(index_name)
    except Exception as e:
        print(f"ERROR connecting to index: {e}")
        return
        
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Generating embeddings for {len(ADVICE_CATALOG)} advice entries...")
    vectors_to_upsert = []
    
    for i, item in enumerate(ADVICE_CATALOG):
        text = item["text"]
        category = item["category"]
        
        # Generate 384-dimensional embedding
        embedding = model.encode(text).tolist()
        
        vectors_to_upsert.append({
            "id": f"advice_{i}",
            "values": embedding,
            "metadata": {
                "text": text,
                "category": category
            }
        })
        
    print("Upserting vectors into Pinecone...")
    index.upsert(vectors=vectors_to_upsert)
    print("Successfully populated the Pinecone Vector Database!")

if __name__ == "__main__":
    upsert_to_pinecone()
