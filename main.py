import PyPDF2
import os
import torch
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()
genai_api_key = os.getenv("GEMINI_API_KEY") or "AIzaSyC87pJSv_ocfWQoKgdSP7aeNT3kYxSDguk"
genai.configure(api_key=genai_api_key)  
gemini_model = genai.GenerativeModel("gemini-2.0-pro-exp")
 
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()
 
def generate_embeddings(text):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding = model.encode([text])
    dimension = embedding.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embedding, dtype=np.float32))
   
    return index, embedding, text
 
 
def summarize_text(text):
    response = gemini_model.generate_content(f"Summarize the following document:\n{text}")
    return response.text
 
 
st.title("Get Your Summary")
 
uploaded_file = st.file_uploader("Upload Here", type="pdf")
 
if uploaded_file:
    st.write("Processing file...")
 
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Text", pdf_text[:1000], height=300)
    index, embedding, stored_text = generate_embeddings(pdf_text)
    D, I = index.search(np.array(embedding, dtype=np.float32), 1)
    retrieved_text = stored_text
    summary = summarize_text(retrieved_text)
    st.subheader("Summary")
    st.write(summary)
 
st.info("Developed using Streamlit, FAISS, and Google Gemini LLM")
 