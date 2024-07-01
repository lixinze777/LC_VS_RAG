import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex
 

api_key = "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"
os.environ["OPENAI_API_KEY"] = api_key
model_name = "gpt-4o-2024-05-13"

reader = SimpleDirectoryReader(input_files=["sample.txt"])
document = reader.load_data()

parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)
nodes = parser.get_nodes_from_documents(document)
index = VectorStoreIndex(nodes)
print(index)


query_engine = index.as_query_engine()
response = query_engine.query("Summarize the text")
print(response)