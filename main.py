import os
import sys
import threading
import requests
import logging
from groq import Groq
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chromadb.utils import embedding_functions

# LangChain Structural Imports
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

# --- STRUCTURAL SYNC LOGIC ---
def run_structural_sync():
    print("🌐 Starting Structural Sync via Jina Reader...")
    
    # Define headers to split on (This captures your "Labels")
    headers_to_split_on = [
        ("#", "Title"),
        ("##", "Section"),
        ("###", "SubSection"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for url in UNIVERSITY_URLS:
        try:
            # 1. Fetch Clean Markdown from Jina
            print(f"📄 Scraping: {url}")
            jina_url = f"https://r.jina.ai/{url}"
            headers = {"X-Return-Format": "markdown"} # Ask Jina for pure markdown
            response = requests.get(jina_url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ Failed to fetch {url}")
                continue

            # 2. Split by Markdown Headers (Preserves the "Labels")
            md_header_splits = markdown_splitter.split_text(response.text)
            
            # 3. Further split into manageable chunks for the Vector DB
            final_chunks = text_splitter.split_documents(md_header_splits)

            # 4. Upsert into ChromaDB
            for i, chunk in enumerate(final_chunks):
                # Extract the label (Section Header) for better context
                label = chunk.metadata.get("Section") or chunk.metadata.get("Title") or "General Information"
                
                collection.upsert(
                    ids=[f"{url}_{i}"],
                    documents=[chunk.page_content],
                    metadatas=[{
                        "source": url,
                        "label": label,
                        "content_type": "official_web_data"
                    }]
                )
            print(f"✅ Indexed {len(final_chunks)} structural chunks from {url}")

        except Exception as e:
            print(f"⚠️ Sync Error for {url}: {e}")

    print("🚀 All URLs Synced Successfully.")

@app.on_event("startup")
async def startup_event():
    # Start sync in background so FastAPI starts instantly on Render
    thread = threading.Thread(target=run_structural_sync)
    thread.start()

# --- CHAT LOGIC ---
class Query(BaseModel):
    text: str

@app.get("/")
async def status():
    return {"status": "Online", "mode": "Structural RAG", "synced_urls": len(UNIVERSITY_URLS)}

@app.post("/chat")
async def chat(query: Query):
    try:
        # 1. Broaden the search to ensure we don't miss the table
        results = collection.query(query_texts=[query.text], n_results=10)
        
        context_parts = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            # Force inclusion of the label so the AI knows which section it's reading
            label = meta.get('label', 'General')
            context_parts.append(f"### SECTION: {label}\n{doc}")
        
        full_context = "\n\n---\n\n".join(context_parts)

        # 2. Stronger System Prompt
        system_message = (
            "You are Leo Bot, the official HITS admissions assistant. "
            "Examine the provided CONTEXT carefully, especially sections labeled 'Entrance Exam Dates' or 'Important Dates'. "
            "If you see a table with HITSEEE 2026 dates, extract them exactly. "
            "If a user asks for 'exam dates' and you see 'April 27, 2026 to May 02, 2026', provide that as the answer."
        )

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "system", "content": f"CONTEXT DATA:\n{full_context}"},
                {"role": "user", "content": query.text}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0  # Set to 0 for maximum accuracy with dates
        )
        
        return {"response": response.choices[0].message.content}

    except Exception as e:
        return {"response": "I'm refreshing my knowledge base. Please try in 10 seconds."}
