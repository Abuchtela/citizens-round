import streamlit as st
from llama_index import StorageContext, load_index_from_storage
import os
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

# load from disk
#index = GPTVectorStoreIndex.load_from_disk('index.json')
index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage"))
query_engine = index.as_query_engine()

# while True:
question = st.text_input("Question", placeholder="Enter your question here")

if question != "":
    response = query_engine.query(question)
    display = "\n" + "\n**" + question + "**\n" + str(response)
    st.markdown(display)
