import os
import streamlit as st
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

import streamlit as st
from langchain.llms import OpenAI
from llama_index import StorageContext, load_index_from_storage, GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, Prompt

from trubrics.integrations.streamlit import FeedbackCollector

collector = FeedbackCollector()

# set the model and parameters
llm_predictor = LLMPredictor(llm=OpenAI(model_name='text-davinci-003', temperature=0))
service_context = ServiceContext.from_defaults(
  llm_predictor=llm_predictor
)

# custom prompt
template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "If the question includes the name of the grantee, then append their grantee website URL and their Twitter handle at the end of the response.\n"
    "Do not respond to questions that ask to sort or rank grantees. Do not respond to questions that ask to compare \
    grantees. Similarly, do not respond to questions that ask for advise on which grantee to donate contributions to. \
    Few examples of such questions are (a) Which grantee had most impact on Gitcoin? (b) Who should I donate to \
    (c) Rank the grantees by impact (d) Compare work of one grantee versus another \
    For such questions, do not share any grantee information and just say: Dear human, I am told not to influence you with my biases for such queries. \
    The burden of choosing the public greats and saving the future of your kind lies on you. Choose well! \n"
    "If the answer is not available in the context information given above, respond: I don't have an answer for this"
    "Given this information, please answer the question: {query_str}\n" 
)
qa_template = Prompt(template)

# load from disk
#index = GPTVectorStoreIndex.load_from_disk('index.json')
index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage"))
query_engine = index.as_query_engine(service_context=service_context, text_qa_template=qa_template)

st.title("Gitcoin Citizens Round")
st.markdown("Hi there! 👋 [The Gitcoin Citizens Round](https://gov.gitcoin.co/t/rewarding-the-community-gitcoin-citizen-round-1/14905) aims to reward people and grassroots projects \
            that have contributed to Gitcoin’s success, specifically by engaging with the wider ecosystem and community")
st.markdown("Gitcoin Citizens Round #1 is **[live](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc)**! \
            Donations open until June 27th 23:59 UTC")
st.markdown("Below, you'll find some links that can give you more information about the grantees on Explorer. \
            And if you have any questions about the Round or the impact that grantees have made, feel free to ask away! 🙌")

#col1, col2, col3 = st.columns(3)

with st.expander("See links to browse through grantees on Explorer"):
    st.markdown("[40acres DAO](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-57)")
    st.markdown("[All for Climate DAO](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-53)")
    st.markdown("[Archimedes' Lever](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-19)")
    st.markdown("[Bankless Academy](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-16)")
    st.markdown("[Bankless Hindi](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-46)")
    st.markdown("[BanklessDAO's Gitcoin Citizens](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-32)")
    st.markdown("[Biteye](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-25)")
    st.markdown("[Blu3 Global](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-10)")
    st.markdown("[Bob Jiang](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-33)")
    st.markdown("[Borderless](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-15)")
    st.markdown("[Carl Cervone - Onchain Impact Evaluatooooor](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-18)")
    st.markdown("[Carlos Melgar - Community Supportooooooor](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-59)")
    st.markdown("[DeSci Round Operators](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-48)")
    st.markdown("[Dune Analytics dashboard - Proof Of Stake by Vitalik Buterin - Digital Book ](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-39)")
    st.markdown("[Eartbased soul](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-36)")
    st.markdown("[Gitcoin Awareness and Female Founder Amplification](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-51)")
    st.markdown("[Gitcoin Gas Optimization](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-14)")
    st.markdown("[Gitcoin Onboarding with Regens Unite](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-35)")
    st.markdown("[greenpill.network](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-47)")
    st.markdown("[Indigenous Voices in Gitcoin by Mycelia](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-54)")
    st.markdown("[ITU Blockchain](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-45)")
    st.markdown("[Jimi Cohen - Gitcoin Radio](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-52)")
    st.markdown("[Jon Ruth - Discord and Telegram Superhero](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-20)")
    st.markdown("[Kairos Research](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-40)")
    st.markdown("[Keith Comito: Gitcoin Citizen - Building in Public and Bringing Gitcoin to the Masses!](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-31)")  
    st.markdown("[Karma](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-30)")
    st.markdown("[Lefteris Karapetsas](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-34)")
    st.markdown("[Mars - Gitcoin citizen](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-58)")
    st.markdown("[OpenLore Library](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-38)")
    st.markdown("[Owocki/Supermodular.xyz (FBO Gitcoin Matching Pool)](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-49)")
    st.markdown("[Quadratic Trust - Anne Connelly](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-22)")
    st.markdown("[ZER8's Gitcoin Citizen Round Application](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-41)")
    st.markdown("[Zuzalu Gitcoin Hype Squad](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc-9)")

st.markdown("Note: [Gitcoin Explorer](https://explorer.gitcoin.co/#/round/10/0x984e29dcb4286c2d9cbaa2c238afdd8a191eefbc) is the single source of truth for information on the Citizens Round. \
            This prototype is built on LLM technology with known limitations. \
            Refer the grantee information on the above links before making final funding decisions.")

st.markdown("**Not sure what to ask? Try these**: *Tell me what I need to know about the Gitcoin Citizens Round as a donor*,  \
            *Tell me about the impact has <grantee> created*")

question = st.text_input("", placeholder="Enter your question here")

if question != "":
    response = query_engine.query(question)
    display = "\n" + str(response) + "\n"
    st.markdown(display)
    # st.markdown(llm_predictor.last_token_usage)
    #feedback = collector.st_feedback(
    #  feedback_type="thumbs",
    #  path="/logs/thumbs_feedback.json"
    #)

