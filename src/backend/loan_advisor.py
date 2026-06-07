import os
import traceback
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

class PineconeAdviceVectorStore:
    def __init__(self):
        # Resolve .env path from the project root (3 levels up from this file)
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
        print(f"[PineconeStore] Resolved .env path: {env_path}")
        print(f"[PineconeStore] .env file exists: {os.path.exists(env_path)}")
        load_dotenv(dotenv_path=env_path)
        self.api_key = os.getenv('PINECONE_API_KEY')
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'loan-advice-index')
        print(f"[PineconeStore] API key found: {bool(self.api_key)}")
        
        self.pc = None
        self.index = None
        self.model = None
        
        if not self.api_key:
            print("[PineconeStore] WARNING: No API key found, skipping initialization.")
            return

        # Step 1: Connect to Pinecone
        try:
            self.pc = Pinecone(api_key=self.api_key)
            self.index = self.pc.Index(self.index_name)
            print(f"[PineconeStore] Successfully connected to Pinecone index '{self.index_name}'")
        except Exception as e:
            print(f"[PineconeStore] ERROR connecting to Pinecone: {e}")
            traceback.print_exc()
            return  # No point loading model if Pinecone fails

        # Step 2: Load sentence-transformers model (separate try/except)
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[PineconeStore] Successfully loaded SentenceTransformer model.")
        except Exception as e:
            print(f"[PineconeStore] ERROR loading SentenceTransformer model: {e}")
            traceback.print_exc()
                
    def get_advice(self, reasoning_text, top_k=2):
        if not self.index or not self.model:
            return [{"advice": "Vector DB not connected. Please check Pinecone API key."}]
            
        try:
            # Generate embedding for the reasoning text
            query_embedding = self.model.encode(reasoning_text).tolist()
            
            # Search the Pinecone index for semantically similar advice
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            advice_list = []
            for match in results.get('matches', []):
                advice_list.append({
                    "advice": match['metadata'].get('text', ''),
                    "category": match['metadata'].get('category', 'General')
                })
                
            return advice_list
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return [{"advice": "An error occurred while fetching advice from the Vector Database."}]
