import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os

# LangChain & Ollama + PDF processing
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# FastAPI app setup
app = FastAPI()

# CORS setup so frontend can access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now (adjust in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
VECTOR_STORE_DIR = "db"
PDF_PATH = "uploaded.pdf"
vectorstore = None  # Will hold vector store instance

# Request model for queries
class QueryRequest(BaseModel):
    query: str

# Upload PDF endpoint
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with open(PDF_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[INFO] Saved uploaded file as {PDF_PATH}")

        # Process PDF and build vectorstore
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        embeddings = OllamaEmbeddings(model="mistral")
        global vectorstore
        vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=VECTOR_STORE_DIR)
        vectorstore.persist()

        return {"message": f"Uploaded and indexed {file.filename} successfully."}
    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}

# Ask a question to the RAG system
@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        if not os.path.exists(PDF_PATH):
            return {"response": "Please upload a PDF first."}

        global vectorstore
        if vectorstore is None:
            # Load from persisted directory if needed
            embeddings = OllamaEmbeddings(model="mistral")
            vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings)

        llm = Ollama(model="mistral")
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        result = qa.run(request.query)

        return {"response": result}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
