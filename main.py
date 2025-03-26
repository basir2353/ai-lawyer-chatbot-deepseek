import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import re

# Set API Key securely
os.environ["GROQ_API_KEY"] = "your_api_key_here"

# Define models
ollama_model_name = "deepseek-r1:14b"
embeddings = OllamaEmbeddings(model=ollama_model_name)
FAISS_DB_PATH = "vectorstore/db_faiss"
pdfs_directory = "pdfs/"

# Initialize LLM
try:
    llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b")
except Exception as e:
    st.error(f"Error initializing ChatGroq: {str(e)}")

# Streamlit UI Settings
st.set_page_config(page_title="Ask AI Lawyer", page_icon="⚖️", layout="centered")
st.markdown(
    """
    <style>
    body { font-family: Arial, sans-serif; }
    .stApp { background-color: #121212; color: white; }
    .light-theme .stApp { background-color: #f5f5f5; color: black; }
    .header { text-align: center; font-size: 36px; font-weight: bold; }
    .social-icons { text-align: center; margin-top: 20px; }
    .social-icons a { margin: 0 15px; text-decoration: none; font-size: 20px; color: #ffffff; }
    .light-theme .social-icons a { color: #000000; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="header">⚖️ Ask AI Lawyer ⚖️</div>', unsafe_allow_html=True)

# Social Links
st.markdown(
    '<div class="social-icons">'
    '<a href="https://www.linkedin.com/in/your-profile" target="_blank">LinkedIn</a>'
    '<a href="https://github.com/your-profile" target="_blank">GitHub</a>'
    '<a href="https://instagram.com/your-profile" target="_blank">Instagram</a>'
    '</div>',
    unsafe_allow_html=True,
)

# File Upload
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], help="Upload legal documents for AI analysis")
user_query = st.text_area("Enter your prompt:", height=150, placeholder="Ask Anything!")

if st.button("Ask AI Lawyer"):
    if uploaded_file and user_query:
        try:
            file_path = os.path.join(pdfs_directory, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PDFPlumberLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            text_chunks = text_splitter.split_documents(documents)
            faiss_db = FAISS.from_documents(text_chunks, embeddings)
            faiss_db.save_local(FAISS_DB_PATH)
            retrieved_docs = faiss_db.similarity_search(user_query)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            prompt_template = ChatPromptTemplate.from_template("""
                You are an AI Lawyer specializing in legal matters for Pakistan and the United States. 
                Use the provided context to give an accurate, well-structured, and professional legal response. 
                Ensure the answer is relevant to the legal systems of Pakistan and the US. 
                If the context does not contain enough information, state that you cannot provide a definitive answer.
                
                Question: {question} 
                Context: {context} 
                
                Answer:
            """)
            chain = prompt_template | llm_model
            response = chain.invoke({"question": user_query, "context": context})
            
            st.chat_message("user").write(user_query)
            st.chat_message("AI Lawyer").write(response)
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")
    else:
        st.error("Please upload a valid PDF and enter a question!")
