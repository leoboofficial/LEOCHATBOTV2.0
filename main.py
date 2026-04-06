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
        # Check if DB has data
        if collection.count() == 0:
            return {"response": "I am currently performing a structural sync of the HITS website. Please wait a moment."}

        # 1. Retrieve the top 6 most relevant structural chunks
        results = collection.query(query_texts=[query.text], n_results=6)
        
        # 2. Build the context string including the Labels
        context_parts = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            context_parts.append(f"[Section: {meta['label']}]\n{doc}")
        
        full_context = "\n\n---\n\n".join(context_parts)

        # 3. Generate response with Llama 3.3
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are Leo Bot, the official HITS expert. "
                        "Use the following structured website data to answer. "
                        "If a date or fee is in a table, report it accurately. "
                        "If the answer isn't in the context, say you don't know.\n\n"
                        f"CONTEXT:\n{full_context}"
                    )
                },
                {"role": "user", "content": query.text}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        
        return {"response": response.choices[0].message.content}

    except Exception as e:
        print(f"Chat Error: {e}")
        return {"response": "The knowledge base is currently refreshing. Please try again in 30 seconds."}
