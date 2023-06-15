import streamlit as st
from llama_index import StorageContext, load_index_from_storage
import os
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor

llm_predictor = LLMPredictor(llm=OpenAI(model_name='gpt-3.5-turbo', temperature=0))
service_context = ServiceContext.from_defaults(
  llm_predictor=llm_predictor
)

# load from disk
#index = GPTVectorStoreIndex.load_from_disk('index.json')
index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage"))
query_engine = index.as_query_engine(service_context=service_context)

st.title("Gitcoin Citizens Round")
st.markdown("[The Gitcoin Citizens Round](https://gov.gitcoin.co/t/rewarding-the-community-gitcoin-citizen-round-1/14905) aims to reward people and grassroots projects \
            that have contributed to Gitcoin’s success, specifically by engaging with the wider ecosystem and community. \
            An umbrella term could be ‘Gitcoin Community Engagement’. More specifically, we aim to reward work contributing directly to Gitcoin’s \
            ‘most important things’ and our Purpose and Essential Intents. For example, contributions in the areas of educational content, data analysis, \
            and support-oriented initiatives.")
st.markdown("Gitcoin Citizens Round #1 is **[live](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc)**! \
            Donations open until June 27th 23:59 UTC")

# while True:
st.markdown("**Learn more about the round and the grantees by asking your questions below:**")
question = st.text_input("", placeholder="Enter your question here")

if question != "":
    response = query_engine.query(question)
    display = "\n" + str(response) + "\n"
    st.markdown(display)
    st.markdown(llm_predictor.last_token_usage)
