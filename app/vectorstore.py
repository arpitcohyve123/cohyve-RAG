# vectorstore.py
import os
from langchain.vectorstores import FAISS
from langchain.schema import Document
from embedder import load_embedder

DB_DIR = "faiss_db"

def build_vectorstore(chunks):
    documents = [
        Document(
            page_content=chunk["content"],
            metadata=chunk["metadata"]
        )
        for chunk in chunks
    ]
    embedder = load_embedder()
    vectorstore = FAISS.from_documents(documents, embedder)
    
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    vectorstore.save_local(DB_DIR)
    print(f"Vectorstore saved to {DB_DIR}")

def load_vectorstore():
    embedder = load_embedder()
    return FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
