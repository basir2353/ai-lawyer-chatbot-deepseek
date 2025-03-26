import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")

# Step 1: Setup LLM (Use DeepSeek R1 with Groq)
llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=api_key)

# Step 2: Retrieve Docs
def retrieve_docs(query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

# Step 3: Answer Question
custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything out of the given context.
Question: {question} 
Context: {context} 
Answer:
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})
