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
os.environ["GROQ_API_KEY"] = "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"

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

# App Icon and Name
st.set_page_config(page_title="AI Legal Assistant", page_icon="⚖️")

# Header with links
st.markdown(
    """
    <h1 style='text-align: center;'>⚖️ AI Legal Assistant</h1>
    <p style='text-align: center;'>
        <a href='https://www.linkedin.com/' target='_blank'>LinkedIn</a> |
        <a href='https://github.com/' target='_blank'>GitHub</a> |
        <a href='https://www.instagram.com/' target='_blank'>Instagram</a>
    </p>
    """,
    unsafe_allow_html=True
)

# Prompt template
custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't make up an answer.
Don't provide anything out of the given context.
Question: {question} 
Context: {context} 
Answer:
"""

def extract_think_section(response_text):
    if not isinstance(response_text, str):
        return "Invalid response format. Could not extract relevant information."
    match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    if match:
        extracted_text = match.group(1).strip()
        extracted_text = extracted_text.replace("\\n", " ")  # Remove newline characters
        sections = extracted_text.split(". ")
        formatted_response = f"### Answer\n\n{sections[0]}\n\n"  # Heading for answer
        if len(sections) > 1:
            formatted_response += "\n".join(sections[1:])  # Separate details into new lines
        return formatted_response
    return "No relevant information found."

def upload_pdf(file):
    try:
        with open(os.path.join(pdfs_directory, file.name), "wb") as f:
            f.write(file.getbuffer())
    except Exception as e:
        st.error(f"Error uploading PDF: {str(e)}")

def load_pdf(file_path):
    try:
        loader = PDFPlumberLoader(file_path)
        return loader.load()
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return []

def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return text_splitter.split_documents(documents)

def create_vector_store(db_path, text_chunks, model_name):
    try:
        faiss_db = FAISS.from_documents(text_chunks, OllamaEmbeddings(model=model_name))
        faiss_db.save_local(db_path)
        return faiss_db
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def retrieve_docs(faiss_db, query):
    return faiss_db.similarity_search(query) if faiss_db else []

def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    try:
        full_response = chain.invoke({"question": query, "context": context})
        if isinstance(full_response, dict) and "text" in full_response:
            full_response = full_response["text"]
        elif not isinstance(full_response, str):
            full_response = str(full_response)
        return extract_think_section(full_response)
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit UI
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
user_query = st.text_area("Enter your prompt:", height=150, placeholder="Ask Anything!")
if st.button("Ask AI Lawyer"):
    if uploaded_file and user_query:
        upload_pdf(uploaded_file)
        documents = load_pdf(os.path.join(pdfs_directory, uploaded_file.name))
        if documents:
            text_chunks = create_chunks(documents)
            faiss_db = create_vector_store(FAISS_DB_PATH, text_chunks, ollama_model_name)
            retrieved_docs = retrieve_docs(faiss_db, user_query)
            response = answer_query(retrieved_docs, llm_model, user_query)
            st.chat_message("user").write(user_query)
            st.chat_message("AI Lawyer").write(response)
    else:
        st.error("Please upload a valid PDF and enter a question!")
