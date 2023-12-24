#!/usr/bin/env python
# coding: utf-8

from dotenv import load_dotenv

import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "SECRET_KEY"

def main():
    st.set_page_config(page_title="LLM Project")
    st.header("LLM Project: Upload your PDF")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    hide_streamlit_style = """
            <style>
            
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=7000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()

        knowledgeBase = FAISS.from_texts(chunks, embeddings)

        userQuestion = st.text_input("Ask a question:")
        onClick = st.button("Generate Q&A")

        if onClick:
            docs = knowledgeBase.similarity_search(userQuestion)

            llm = OpenAI(model="babbage-002")

            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=userQuestion)
                print(cost)

            st.write(response)

if __name__ == '__main__':
    main()