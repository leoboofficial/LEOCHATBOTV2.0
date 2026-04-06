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

# --- ENHANCED STRUCTURAL SYNC ---
def run_structural_sync():
    print("🌐 Starting High-Precision Structural Sync...")
    
    headers_to_split_on = [
        ("#", "Title"),
        ("##", "Section"),
        ("###", "SubSection"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

    for url in UNIVERSITY_URLS:
        try:
            print(f"📄 Scraping: {url}")
            jina_url = f"https://r.jina.ai/{url}"
            # Adding a browser-like header to ensure Jina gets the full rendered table
            headers = {"X-Return-Format": "markdown", "X-Wait-For": "3"} 
            response = requests.get(jina_url, headers=headers, timeout=40)
            
            if response.status_code != 200: continue

            md_header_splits = markdown_splitter.split_text(response.text)
            final_chunks = text_splitter.split_documents(md_header_splits)

            for i, chunk in enumerate(final_chunks):
                label = chunk.metadata.get("Section") or chunk.metadata.get("Title") or "General"
                
                # IMPORTANT: Tag chunks containing dates or from the apply page as high priority
                is_priority = 1 if "apply" in url or "date" in label.lower() or "exam" in label.lower() else 0
                
                collection.upsert(
                    ids=[f"{url}_{i}"],
                    documents=[chunk.page_content],
                    metadatas=[{
                        "source": url,
                        "label": label,
                        "priority": is_priority
                    }]
                )
            print(f"✅ Indexed {url} (Priority: {is_priority})")

        except Exception as e:
            print(f"⚠️ Sync Error: {e}")

@app.on_event("startup")
async def startup_event():
    thread = threading.Thread(target=run_structural_sync)
    thread.start()

class Query(BaseModel):
    text: str

# --- REFINED CHAT LOGIC ---
@app.post("/chat")
async def chat(query: Query):
    try:
        # 1. Search for a larger set of results (n=12) to catch the full table
        results = collection.query(query_texts=[query.text], n_results=12)
        
        context_parts = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            # Wrap each chunk in a clear structural header for the LLM
            context_parts.append(f"### DATA FROM SECTION: {meta['label']}\n{doc}")
        
        full_context = "\n\n---\n\n".join(context_parts)

        # 2. Expert System Prompt
        system_message = (
            "You are Leo Bot, the HITS Admissions Expert. "
            "The current year is 2026. Use the provided context to answer. "
            "IF the user asks for 'exam dates' or 'deadlines', LOOK FOR A TABLE in the context. "
            "IMPORTANT: According to the official HITS schedule in your context: "
            "- Last Date to Apply: April 22, 2026. "
            "- Entrance Exam Dates: April 27, 2026 to May 02, 2026. "
            "Always prioritize these specific dates if they appear in the data."
        )

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"CONTEXT DATA:\n{full_context}\n\nUSER QUESTION: {query.text}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0
        )
        
        return {"response": response.choices[0].message.content}

    except Exception as e:
        return {"response": "System is busy indexing. Please retry in 10 seconds."}
