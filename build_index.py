import os
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from IPython.display import Markdown, display
from llama_index import StorageContext, load_index_from_disk

documents = SimpleDirectoryReader('data').load_data()
index = GPTVectorStoreIndex.from_documents(documents)

# save to disk
#index.save_to_disk('index.json')
index.storage_context.persist(persist_dir="./storage")

# load from disk
loaded_index = load_index_from_disk(StorageContext.from_defaults(persist_dir="./storage"))

response = loaded_index.query("What is Citizens Round?")
print(response)
