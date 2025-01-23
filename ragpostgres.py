import os
import torch
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
import json
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import DictCursor

from dotenv import load_dotenv
import os

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}



def get_db_connection():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=DictCursor)

# Models for FastAPI Request/Response cycle
class QueryRequest(BaseModel):
    user_id: str
    query: str

class QueryResponse(BaseModel):
    response: str

class SessionHistoryResponse(BaseModel):
    user_id: str
    chat_history: List[dict]

class ChatMessage:
    def __init__(self, user_id: str, query: str, response: str, timestamp: datetime = None):
        self.user_id = user_id
        self.query = query
        self.response = response
        self.timestamp = timestamp or datetime.now()

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "query": self.query,
            "response": self.response,
            "timestamp": self.timestamp.isoformat()
        }

class UserSession:
    def __init__(self, user_id: str, max_history: int = 5):
        self.user_id = user_id
        self.max_history = max_history

    def get_history(self) -> List[ChatMessage]:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT query, response, timestamp FROM chat_history WHERE user_id = %s ORDER BY timestamp DESC LIMIT %s",
                    (self.user_id, self.max_history)
                )
                rows = cur.fetchall()                
                return [ChatMessage(self.user_id, row['query'], row['response'], row['timestamp']) for row in rows]

    def add_message(self, query: str, response: str):
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_history (user_id, query, response) VALUES (%s, %s, %s)",
                    (self.user_id, query, response)
                )
                conn.commit()

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, UserSession] = {}

    def get_or_create_session(self, user_id: str) -> UserSession:
        if not user_id:
            user_id = "prem121002@gmail.com"
        if user_id not in self.sessions:
            self.sessions[user_id] = UserSession(user_id)
        return self.sessions[user_id]

class QASystem:
    def __init__(self, pdf_path: str):
        self.session_manager = SessionManager()
        self.setup_qa_system(pdf_path)
    
    def setup_qa_system(self, pdf_path: str):
        try:
            self.llm = OllamaLLM(
                model="llama3.2:1b",
                temperature=0.2,
                gpu=True
            )
        except Exception as e:
            print(f"Error initializing Ollama: {e}")
            raise e

        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
        except Exception as e:
            print(f"Error loading PDF: {e}")
            raise e

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.db = FAISS.from_documents(texts, self.embeddings)

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(),
        )

    def get_response(self, user_id: str, query: str) -> str:
        session = self.get_session(user_id)
        
        try:
            retrieved_context = self.db.as_retriever().get_relevant_documents(query)
            response = self.qa.run({"query": query, "retrieved_context": retrieved_context})
            
            session.add_message(query, response)
            return response
        except Exception as e:
            error_msg = f"Error during QA: {e}"
            return error_msg

    def get_session(self, user_id: str) -> UserSession:
        return self.session_manager.get_or_create_session(user_id)

app = FastAPI()

qa_system = QASystem(pdf_path="D:\RAGProject\CTC related FAQ's.pdf")

@app.post("/start_session/")
def start_session(user_id: str):
    session = qa_system.get_session(user_id)
    return {"message": f"Session started for user: {user_id}"}

@app.post("/ask_question/", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        response = qa_system.get_response(request.user_id, request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/get_history/", response_model=SessionHistoryResponse)
def get_history(user_id: str):
    session = qa_system.get_session(user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    chat_history = [msg.to_dict() for msg in session.get_history()]
    return {"user_id": user_id, "chat_history": chat_history}

@app.delete("/end_session/")
def end_session(user_id: str):
    qa_system.session_manager.sessions.pop(user_id, None)
    return {"message": f"Session for user_id: {user_id} has been ended"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
