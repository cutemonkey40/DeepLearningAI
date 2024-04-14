import streamlit as st
import pandas as pd
import numpy as np 
import time
import anthropic
from io import StringIO
from langchain.chat_models import ChatOpenAI

#text splitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI

# Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
from langchain.vectorstores import FAISS  #facebook vectorizationfrom langchain.chains.question_answering import load_qa_chain
from langchain.chains.question_answering import load_qa_chain

#vectorize db index with chromadb
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

#TXT loader
from langchain.document_loaders import TextLoader  #for textfiles

#URL loader
from langchain.document_loaders import UnstructuredURLLoader  #load urls into docoument-loader

#Selenium URL Loader
from langchain.document_loaders import SeleniumURLLoader

#CSV loader
import csv
from langchain.document_loaders.csv_loader import CSVLoader

#PDF loader
from langchain.document_loaders import UnstructuredPDFLoader

#Directory Loader
from langchain.document_loaders import DirectoryLoader


from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


from langchain.chains import ConversationalRetrievalChain



### Langsmith
from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset

from langchain_community.chat_models import ChatOpenAI



with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="file_qa_api_key:", type="password")   

st.title("📝 File Q&A with Langchain")

uploaded_file = st.file_uploader("Choose a file")

#uploaded_file = st.file_uploader("Upload an article", type=("txt", "md", "pdf"))

if uploaded_file is not None:
    # To read file as bytes:
    #bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)
    article = uploaded_file.read().decode("latin-1")
    #To convert to a string based IO:


question = st.text_input(
    "Ask something about Taiwan Movie Industry:",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

chunk_size = 250
chunk_overlap = 10
temperature= 0.0

if uploaded_file and question and not openai_api_key:
    st.info("Please add your openai_api_key API key to continue.")

if uploaded_file and question and  openai_api_key:
    documents=[]
    loader = UnstructuredPDFLoader(article)
    documents.extend(loader.load())

    #split the documents to textchunk

    text_splitter=CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents =text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="D:/vectorstore/")
    vectordb.persist()
    chain = RetrievalQA.from_chain_type(llm=OpenAI(temperature = temperature ,model_name="gpt-4"),
                     retriever=vectordb.as_retriever(search_kwargs={'k':1}),
                    return_source_documents= True)


    result = chain({'query': question})


    st.write("### Answer")
    st.write(result['result'])



