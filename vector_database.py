from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

pdfs_directory = "pdfs/"

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

file_path = 'pdfs/new.pdf'
documents = load_pdf(file_path)
#print("PDF pages: ",len(documents))

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

file_path = "pdfs/new.pdf"
documents = load_pdf(file_path)
text_chunks = create_chunks(documents)

# print("Text Chunks:", len(text_chunks))



ollama_model_name="deepseek-r1:1.5b"

def  get_embedding_model(ollama_model_name):
    embeddings = OllamaEmbeddings(model=ollama_model_name)
    return embeddings

FAISS_DB_PATH="vectorstore/db_faiss"
faiss_db= FAISS.from_documents(text_chunks,get_embedding_model(ollama_model_name))
faiss_db.save_local(FAISS_DB_PATH)