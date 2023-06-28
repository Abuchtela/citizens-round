import os
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from IPython.display import Markdown, display
from llama_index import StorageContext, load_index_from_storage

# Set the OPENAI_API_KEY environment variable using the value from st.secrets['OPENAI_API_KEY']
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Load documents from the 'data' directory
documents = SimpleDirectoryReader('data').load_data()

# Create an index from the loaded documents
index = GPTVectorStoreIndex.from_documents(documents)

# Save the index to disk
index.storage_context.persist(persist_dir="./storage")

# Load the index from disk for testing
# loaded_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage"))

# Create a query engine from the loaded index
# query_engine = loaded_index.as_query_engine()

# Perform a query using the query engine
# response = query_engine.query("What is Citizens Round?")
# print(response)
