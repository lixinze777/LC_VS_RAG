import os
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
import logging
import sys

# Set up logging to file
log_file = 'app.log'
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s %(message)s')

# Remove the default stream handler
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Add the file handler
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s %(message)s')
file_handler.setFormatter(formatter)
logging.root.addHandler(file_handler)

PERSIST_DIR = "./storage"

api_key = "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"
os.environ["OPENAI_API_KEY"] = api_key
model_name = "gpt-4o-2024-05-13"
embed_name = "text-embedding-3-small"

Settings.embed_model = OpenAIEmbedding(model = embed_name)

reader = SimpleDirectoryReader(input_files=["sample.txt"])
document = reader.load_data()

parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)
nodes = parser.get_nodes_from_documents(document)
index = VectorStoreIndex(nodes)
index.storage_context.persist()

query_engine = index.as_query_engine(model_name=model_name)
response = query_engine.query("Summarize the text")
print(response)

