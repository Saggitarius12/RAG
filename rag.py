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
from typing import List,Dict,Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
    def __init__(self,user_id:str,query:str,response:str,timestamp:datetime=None):
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
    def __init__(self,user_id:str,max_history:int=5):
           self.user_id = user_id
           self.max_history = max_history
           self.chat_history:List[ChatMessage] = []

    def add_message(self,query:str,response:str):
        message = ChatMessage(self.user_id,query,response)
        self.chat_history.append(message)

        if(len(self.chat_history) > self.max_history):
            self.chat_history = self.chat_history[-self.max_history:]

    def get_history(self)->List[ChatMessage]:
        return self.chat_history

class SessionManager:
    def __init__(self):
        self.sessions:Dict[str,UserSession] = {}
        self.history_file = "chat_history.json"
        self.load_history()

    def get_or_create_session(self,user_id:str)->UserSession:
        if not user_id:
            user_id = "prem12"
        if user_id not in self.sessions:
            self.sessions[user_id] = UserSession(user_id)
        return self.sessions[user_id]

    def save_history(self):
        history_data = {}
        for user_id, session in self.sessions.items():
            history_data[user_id] = [msg.to_dict() for msg in session.chat_history]
        
        with open(self.history_file, 'w') as f:
            json.dump(history_data, f)
    
    def load_history(self):
        if not os.path.exists(self.history_file):
            return
        
        try:            
            with open(self.history_file,'r') as file:
                history_data = json.load(file)
                for user_id, messages in history_data.items():
                    session = UserSession(user_id)
                    for msg in messages:
                        session.add_message(msg['query'],msg['response'])
                    self.sessions[user_id] = session

        except Exception as e:
            print(f"Error loading chat history: {e}")

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
            print(documents[0])
        except Exception as e:
            print(f"Error loading PDF: {e}")
            raise e

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)  

        print(texts[0])             

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )        

        self.db = FAISS.from_documents(texts, self.embeddings)
        

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(),            
        )

    def get_response(self, user_id: str, query: str, relevance_threshold: float = 0.3) -> str:
        session = self.get_session(user_id)
        
        try:
            
            top_chunks = self.db.as_retriever().get_relevant_documents(query)
            filtered_chunks = [
                chunk for chunk in top_chunks
                if chunk.metadata.get("similarity_score", 1) >= relevance_threshold
            ] 
            chat_history = "\n".join(
                [f"User: {msg.query}\nAssistant: {msg.response}" for msg in session.get_history()]
            )

            if not filtered_chunks:
                if query.lower() in ["hello","hi","hey"]:
                    print("Hello!, How can I help you today?")
                else:
                    print("No relevant content found. Generating a default response...")
                    response = self.llm.invoke({"query": query})["result"]
                
            else:
                retrieved_context = "\n\n".join(chunk.page_content for chunk in filtered_chunks)
                for i, chunk in enumerate(filtered_chunks):
                    print(f"Chunk {i+1}: {chunk.page_content[:200]}..")
                    print(f"Source Metadata: {chunk.metadata}\n")

                context = (
                    f"Chat History:\n{chat_history}\n\n"
                    f"Retrieved Context:\n{retrieved_context}\n\n"
                    f"Question: {query}\n\n"
                )
                
                result = self.qa.invoke({"query": context})
                response = result["result"]
                        
            session.add_message(query, response)
            self.session_manager.save_history()
            
            return response
        
        except Exception as e:
            error_msg = f"Error during QA: {e}"
            response = error_msg
            return response    
        

    def get_session(self, user_id: str) -> UserSession:
        return self.session_manager.get_or_create_session(user_id)
    
    def calculate_relevancy_score(responses,query):
        relevant_responses = sum([1 for response in responses if query.lower() in response.lower()])

        return relevant_responses/len(responses)

app = FastAPI()

qa_system = QASystem(pdf_path="CTC related FAQ's.pdf")
@app.post("/start_session/")
def start_session(user_id: str):
    session = qa_system.get_session(user_id)
    return {"message": f"Session started for user:{user_id}"}

@app.post("/ask_question/",response_model=QueryResponse)
def ask_question(request:QueryRequest):
    try:
        response = qa_system.get_response(request.user_id, request.query)
        return {"response":response}
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Error processing query:{str(e)}")
    
@app.get("/get_history/", response_model=SessionHistoryResponse)
def get_history(user_id: str):
    session = qa_system.get_session(user_id)
    if not session:
        raise HTTPException(status_code=404,detail="Session not found")
    chat_history = [msg.to_dict() for msg in session.get_history()]
    return {"user_id": user_id, "chat_history": chat_history}

@app.delete("/end_session/")
def end_session(user_id: str):
    qa_system.session_manager.sessions.pop(user_id, None)
    return {"message": f"Session for user_id: {user_id} has been ended"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)