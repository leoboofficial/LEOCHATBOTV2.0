import os
import sys
import threading
import requests
from groq import Groq
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chromadb.utils import embedding_functions
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# --- RENDER SQLITE FIX ---
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass 

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- CONFIGURATION ---
UNIVERSITY_URLS = [
    "https://apply.hindustanuniv.ac.in/hitseee",
    "https://hindustanuniv.ac.in/bachelor-of-technology-btech-aeronautical-engineering/",
    "https://hindustanuniv.ac.in/bachelor-of-technology-btech-aerospace-engineering/"
]

# --- AI & DB INITIALIZATION ---
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
db_client = chromadb.PersistentClient(path="./hits_structural_db")
ef = embedding_functions.DefaultEmbeddingFunction()
collection = db_client.get_or_create_collection(name="hits_web_data", embedding_function=ef)

# --- HIGH-PRECISION STRUCTURAL SYNC ---
def run_structural_sync():
    print("🌐 Starting High-Priority Structural Sync...")
    
    headers_to_split_on = [("#", "Title"), ("##", "Section"), ("###", "SubSection")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

    for url in UNIVERSITY_URLS:
        try:
            print(f"📄 Scraping: {url}")
            jina_url = f"https://r.jina.ai/{url}"
            headers = {"X-Return-Format": "markdown", "X-Wait-For": "2"} 
            response = requests.get(jina_url, headers=headers, timeout=40)
            
            if response.status_code != 200: continue

            md_header_splits = markdown_splitter.split_text(response.text)
            final_chunks = text_splitter.split_documents(md_header_splits)

            for i, chunk in enumerate(final_chunks):
                label = chunk.metadata.get("Section") or chunk.metadata.get("Title") or "General"
                
                # --- NEW PRIORITY LOGIC ---
                # Tags HITSEEE, Aeronautical, and Aerospace as Priority 1
                priority_keywords = ["apply", "aeronautical", "aerospace"]
                is_priority = 1 if any(kw in url.lower() for kw in priority_keywords) or "date" in label.lower() else 0
                
                collection.upsert(
                    ids=[f"{url}_{i}"],
                    documents=[chunk.page_content],
                    metadatas=[{
                        "source": url,
                        "label": label,
                        "priority": is_priority
                    }]
                )
            print(f"✅ Indexed {url} with Priority: {is_priority}")

        except Exception as e:
            print(f"⚠️ Sync Error: {e}")

@app.on_event("startup")
async def startup_event():
    thread = threading.Thread(target=run_structural_sync)
    thread.start()

class Query(BaseModel):
    text: str

# --- CHAT LOGIC ---
@app.post("/chat")
async def chat(query: Query):
    try:
        # Increase results to ensure we get data from all 3 priority URLs
        results = collection.query(query_texts=[query.text], n_results=12)
        
        context_parts = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            context_parts.append(f"### SOURCE [{meta['label']}]:\n{doc}")
        
        full_context = "\n\n---\n\n".join(context_parts)

        system_message = (
            "You are Leo Bot, the HITS Academic Expert. "
            "Use the provided context to answer questions about Admissions and Aero-Science. "
            "If the question is about labs, faculty, or alumni in Aeronautical/Aerospace, look for specific mentions in the context. "
            "If the answer isn't there, suggest checking the official HITS department brochures."
        )

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"CONTEXT:\n{full_context}\n\nUSER QUESTION: {query.text}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0
        )
        return {"response": response.choices[0].message.content}

    except Exception as e:
        return {"response": "Syncing new department data. Please wait 10 seconds."}
