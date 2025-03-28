# import PyPDF2
# import os
# import torch
# import faiss
# import numpy as np
# import streamlit as st
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
# import google.generativeai as genai

# # Load API Key
# load_dotenv()
# genai_api_key = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=genai_api_key)  
# gemini_model = genai.GenerativeModel("gemini-2.0-pro-exp")

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_file):
#     reader = PyPDF2.PdfReader(pdf_file)
#     text = ""
#     for page in reader.pages:
#         extracted = page.extract_text()
#         if extracted:
#             text += extracted + "\n"
#     return text.strip()

# # Function to generate embeddings
# def generate_embeddings(text):
#     model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#     embedding = model.encode([text])
#     dimension = embedding.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(np.array(embedding, dtype=np.float32))
#     return index, embedding, text

# # Function to get AI response
# def get_ai_response(question, context):
#     response = gemini_model.generate_content(f"Answer the following question based on the document:\n\nContext: {context}\n\nQuestion: {question}")
#     return response.text

# # Streamlit UI
# st.title("Get the doubts clarified with your PDF")

# # Upload PDF
# uploaded_file = st.file_uploader("Upload Your PDF", type="pdf")

# if uploaded_file:
#     st.write(f"Processing  your {uploaded_file.name} file...")
#     pdf_text = extract_text_from_pdf(uploaded_file)
#     index, embedding, stored_text = generate_embeddings(pdf_text)
#     st.session_state["stored_text"] = stored_text  # Store extracted text in session state
#     st.success("PDF Uploaded and Processed Successfully!")

# # Chat Area
# if "stored_text" in st.session_state:
#     st.subheader("Ask Questions About Your PDF")
#     user_input = st.text_input("Enter your question:")
    
#     if user_input:
#         response = get_ai_response(user_input, st.session_state["stored_text"])
#         st.write("**Answer:**", response)
    
#     # Continue or Exit
#     if st.button("Exit Chat"):
#         st.session_state.clear()
#         st.write("Chat session ended. Refresh to start over.")
import PyPDF2
import os
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load API Key
load_dotenv()
genai_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=genai_api_key)  
gemini_model = genai.GenerativeModel("gemini-2.0-pro-exp")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()

# Function to generate embeddings
def generate_embeddings(text):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding = model.encode([text])
    dimension = embedding.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embedding, dtype=np.float32))
    return index, embedding, text

# Function to get AI response
def get_ai_response(question, context):
    response = gemini_model.generate_content(f"Answer the following question based on the document:\n\nContext: {context}\n\nQuestion: {question}")
    return response.text

# Streamlit UI
st.set_page_config(page_title="PDF Chat Assistant", page_icon="📄", layout="wide")
st.title("Get clarify your doubts with your PDF")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "stored_text" not in st.session_state:
    st.session_state.stored_text = None

# Upload PDF
uploaded_file = st.file_uploader("Upload Your PDF", type="pdf", key="pdf_uploader")

if uploaded_file:
    st.subheader(f"Processing: **{uploaded_file.name}**...")
    pdf_text = extract_text_from_pdf(uploaded_file)
    index, embedding, stored_text = generate_embeddings(pdf_text)
    
    # Store extracted text and reset chat history when a new PDF is uploaded
    st.session_state.stored_text = stored_text
    st.session_state.chat_history = []
    
    st.success("PDF Uploaded and Processed Successfully!")

# Chat Area
if st.session_state.stored_text:
    st.subheader("Ask your doubt and get clarified: ")

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["response"])

    # User Input
    user_input = st.text_input("Type your question and press Enter:", key="user_input")
    
    if user_input:
        response = get_ai_response(user_input, st.session_state.stored_text)
        
        # Append to chat history
        st.session_state.chat_history.append({"question": user_input, "response": response})

        # Display latest response
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.write(response)

# Exit Chat Button - Resets the session
if st.button("Exit Chat"):
    st.session_state.clear()  # Clears everything including uploaded file
    # st.session_state["pdf_uploader"] = None  # Ensures file_uploader resets
    st.rerun()  # Refresh the app

