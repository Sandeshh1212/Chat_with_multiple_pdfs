import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pickle
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    # print("text")
    return text

def get_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000 , chunk_overlap= 1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectors(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model= "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks,embedding = embeddings)
    vector_store.save_local("faiss_index")
    print("faiss index created successfully")

def get_conversational_chain():
    prompt_template = """ 
    Answer the question as detailed as possible from the provided context , make sure to provide all the details , 
    if the answer is not in the provided context just "say answer is not available in the context" don't provide the wrong answer
    Context : \n{context}?\n
    Question : \n{question}\n

    Answer : 

    """

    model = ChatGoogleGenerativeAI(model="gemini-pro" , temperature=0.3)
    prompt = PromptTemplate(template= prompt_template, input_variables=  ["context","question"])
    chain = load_qa_chain(model,chain_type="stuff",prompt= prompt)
    return chain

def get_user_input(user_ques):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_ques)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents" : docs , "question":user_ques },
        return_only_outputs= True)
    
    print(response)
    st.write("Reply : ",response["output_text"])




def main():
    st.set_page_config("Chat with Multiple PDF")
    st.header("Chat with Multiple PDF using Gemini ")
    # st.set_option('server.allow_dangerous_deserialization', True)
    #


    user_question = st.text_input("Ask a question from PDF files ")
     
    if user_question:
        get_user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload Multiple PDFs here", type="pdf", accept_multiple_files=True)
        if st.button("Submit and process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_chunk(raw_text)
                get_vectors(chunks)
                st.success("Done")

            
if __name__ == "__main__":
    # st.set_option('server.allow_dangerous_deserialization', True)

    main()

