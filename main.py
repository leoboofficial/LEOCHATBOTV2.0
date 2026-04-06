import os
import sys
import threading # New import for background sync
from groq import Groq
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- RENDER SQLITE FIX ---
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass 

app = FastAPI()

# GREETING & URLS
UNIVERSITY_URLS = [
    "https://hindustanuniv.ac.in/bachelor-of-technology-btech-aeronautical-engineering/",
    "https://hindustanuniv.ac.in/bachelor-of-technology-btech-aerospace-engineering/",
    "https://apply.hindustanuniv.ac.in/",
    "https://apply.hindustanuniv.ac.in/hitseee"
]
EXACT_GREETING = "Hello! I am Leo Bot, your Dynamic HITS Expert..."

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- AI & DB INITIALIZATION ---
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
db_client = chromadb.PersistentClient(path="./dynamic_hits_db")
ef = embedding_functions.DefaultEmbeddingFunction()
collection = db_client.get_or_create_collection(name="hits_web_data", embedding_function=ef)

# --- BACKGROUND SYNC FUNCTION ---
def run_sync():
    print("🌐 Background Sync Started...")
    try:
        loader = WebBaseLoader(UNIVERSITY_URLS)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        
        for i, chunk in enumerate(chunks):
            collection.upsert(
                ids=[f"web_chunk_{i}"],
                documents=[chunk.page_content],
                metadatas=[{"source": chunk.metadata['source']}]
            )
        print(f"✅ Sync Successful: {len(chunks)} blocks updated.")
    except Exception as e:
        print(f"⚠️ Sync Error: {e}")

@app.on_event("startup")
async def startup_event():
    # This starts the sync in a separate thread so the server opens INSTANTLY
    thread = threading.Thread(target=run_sync)
    thread.start()

class Query(BaseModel):
    text: str

@app.get("/")
async def status():
    return {"status": "Online", "mode": "Dynamic Sync", "info": "Knowledge base is updating in background."}

@app.post("/chat")
async def chat(query: Query):
    try:
        if query.text.lower().strip() in ["hi", "hello", "hey"]:
            return {"response": EXACT_GREETING}

        # Check if DB is still empty
        if collection.count() == 0:
            return {"response": "I am currently synchronizing with the HITS website. Please give me 1 minute to finish learning!"}

        results = collection.query(query_texts=[query.text], n_results=3)
        
        if results['documents'] and results['distances'][0][0] < 1.7:
            context = "\n".join(results['documents'][0])
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"You are Leo Bot. Use this HITS Website context: {context}"},
                    {"role": "user", "content": query.text}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1
            )
            return {"response": response.choices[0].message.content}
        
        return {"response": "I couldn't find that specific detail. Please check info@hindustanuniv.ac.in."}
    except Exception as e:
        return {"response": "The server is warming up. Please try again in a few moments."}
