import os
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

ffrom llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from IPython.display import Markdown, display
from llama_index import StorageContext, load_index_from_storage

documents = SimpleDirectoryReader('data').load_data()
index = GPTVectorStoreIndex.from_documents(documents)

# save to disk
index.storage_context.persist(persist_dir="./storage")

# load from disk
loaded_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage"))

query_engine = loaded_index.as_query_engine()

response = query_engine.query("What is Citizens Round?")
print(response)

