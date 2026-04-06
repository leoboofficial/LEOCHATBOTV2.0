import os
import sys
import logging
from groq import Groq
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- RENDER SQLITE FIX ---
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass 

app = FastAPI()

# 1. TARGET URLS (Add the most important HITS pages here)
UNIVERSITY_URLS = [
    "https://hindustanuniv.ac.in/aeronautical_engineering.php",
    "https://hindustanuniv.ac.in/admission_office.php",
    "https://hindustanuniv.ac.in/hitseee-2026.php" 
]

EXACT_GREETING = "Hello! I am Leo Bot, your Dynamic HITS Expert..."

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- AI & DB INITIALIZATION ---
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# We use a persistent client so the data stays between restarts
db_client = chromadb.PersistentClient(path="./dynamic_hits_db")
ef = embedding_functions.DefaultEmbeddingFunction()
collection = db_client.get_or_create_collection(name="hits_web_data", embedding_function=ef)

# --- THE "MAGIC" STARTUP SYNC ---
@app.on_event("startup")
async def sync_with_university_website():
    print("🌐 Connecting to HITS Website for Live Sync...")
    try:
        # Load directly from the URLs
        loader = WebBaseLoader(UNIVERSITY_URLS)
        docs = loader.load()
        
        # Split into small pieces so the AI can find specific answers
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)
        
        # Ingest into ChromaDB
        for i, chunk in enumerate(chunks):
            collection.upsert(
                ids=[f"web_chunk_{i}"],
                documents=[chunk.page_content],
                metadatas=[{"source": chunk.metadata['source']}]
            )
        print(f"✅ Sync Successful: {len(chunks)} knowledge blocks updated.")
    except Exception as e:
        print(f"⚠️ Sync Error: {e}. Bot will use last cached data.")

class Query(BaseModel):
    text: str

@app.get("/")
async def status():
    return {"bot": "HITS Dynamic Leo", "status": "active", "synced_urls": UNIVERSITY_URLS}

@app.post("/chat")
async def chat(query: Query):
    try:
        # 1. Quick Hello
        if query.text.lower().strip() in ["hi", "hello", "hey"]:
            return {"response": EXACT_GREETING}

        # 2. Retrieval
        results = collection.query(query_texts=[query.text], n_results=3)
        
        # 3. Generation via Groq
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
        
        return {"response": "That information isn't on the HITS pages I currently track. Please check info@hindustanuniv.ac.in."}

    except Exception as e:
        return {"response": "System is updating its knowledge. Please try in 10 seconds."}
