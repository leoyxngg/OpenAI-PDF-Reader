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

pdf_reader = PdfReader("Proving Program Correctness.pdf")

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

myQuestion = input("What is your question about this document? ")

docs = knowledgeBase.similarity_search(myQuestion)

llm = OpenAI(model="babbage-002")

chain = load_qa_chain(llm, chain_type='stuff')

with get_openai_callback() as cost:
    response = chain.run(input_documents=docs, question=myQuestion)
    print(cost)

print(response)