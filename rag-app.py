__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from streamlit import logger
import sqlite3

import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI 
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

load_dotenv()

app_logger = logger.get_logger('SMI_APP')
app_logger.info(f"sqlite version: {sqlite3.sqlite_version}")
app_logger.info(f"sys version:  {sys.version}")

def extract_text_from_url(url):
  logging.debug("Extracting text from url %s", url)
  response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
  soup = BeautifulSoup(response.content, 'html.parser')
  return soup.get_text()

st.subheader("Ask AI to help you search your URL")

# Number of URLs to capture
num_urls = st.number_input("How many URLs do you want to input?", min_value=1, max_value=10, value=3)

# Initialize an empty list to store URLs
urls = []

# Loop to create input fields for each URL
for i in range(num_urls):
    url = st.text_input(f"Enter URL {i+1}: ", "http://")
    urls.append(url)

logging.debug(f"urls to fetch - {len(urls)}")

# Set OpenAI API key from Streamlit secrets
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set OpenAI key from env
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")

def generate_response(db, input_text):
    # use our vector store to find similar text chunks
    logging.debug(f"searching Chroma")
    results = db.similarity_search(
        query=input_text,
        k=5
    )

    # define the prompt template
    context_template = """
    You are a chat bot who loves to help people! Given the following context sections, answer the
    question using only the given context. If you are unsure and the answer is not
    explicitly written in the documentation, say "Sorry, I don't know how to help with that."

    Context sections:
    {context}

    Question:
    {users_question}

    Answer:
    """

    # define the prompt template to test without the context provided in the knowledge base
    gpt_template = """
    You are a chat bot who loves to help people! Answer the
    question.

    Question:
    {users_question}

    Answer:
    """

    prompt = PromptTemplate(template=context_template, input_variables=["context", "users_question"])

    # fill the prompt template
    prompt_text = prompt.format(context = results, users_question = input_text)

    # ask the defined LLM
    logging.debug(f"about to call OpenAI")
    model = OpenAI(temperature=1,api_key=OPENAI_API_KEY)

    logging.debug(f"LLM Result")
    st.info(model.invoke(prompt_text))


def create_knowledgebase(urls):
    # File to write the URLs
    file_name = "knowledgebase.txt"

    # Open the file in write mode
    logging.debug(f"fetching url and storing into file {file_name}")
    with open(file_name, 'w', encoding='utf-8') as file:
        # Loop through the list of URLs
        for url in urls:
            logging.debug(f"processing url {url}")
            text = extract_text_from_url(url)
            text = text.replace('\n', '')
            file.write(text + "\n")

    # load the document
    logging.debug(f"loading kb")
    with open(f'./{file_name}', encoding='utf-8') as f:
        text = f.read()

    # define the text splitter
    logging.debug(f"splitting kb")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap  = 100,
        length_function = len,
    )

    texts = text_splitter.create_documents([text])

    # define the embeddings model
    logging.debug(f"about to run embedding")
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # use the text chunks and the embeddings model to fill our vector store
    logging.debug(f"loading Chroma")
    db = Chroma.from_documents(texts, embeddings)

    return db


with st.form("my_form"):
    text = st.text_area(
        "Please enter your question",
        "What question would you like answered by the URL?"
    )
    submitted = st.form_submit_button("Ask AI")
    if not OPENAI_API_KEY.startswith("sk-"):
        st.warning("Please enter your OpenAI API key", icon="⚠")
    if submitted and OPENAI_API_KEY.startswith("sk-"):
        generate_response(create_knowledgebase(urls), text)
