import os
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, TreeIndex
from llama_index.readers.string_iterable import StringIterableReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
import logging
import sys
import jsonlines
from tqdm import tqdm
import numpy as np

api_key = "your key here"
os.environ["OPENAI_API_KEY"] = api_key
model_name = "gpt-4o-2024-05-13"
embed_name = "text-embedding-3-small"
Settings.embed_model = OpenAIEmbedding(model = embed_name)

if __name__ == "__main__":
    folder_path = '../../datasets/sample_set_filtered'
    dataset_files = [
        'coursera.jsonl',
        'quality.jsonl',
        'toeflqa.jsonl',
        'novelqa.jsonl'
    ]

    for file in dataset_files:
        # Set up logging to file
        print("Start to process: " + file)
        with jsonlines.open(folder_path + "/" + file) as f:
            jsonlist = list(f)
    
        output_list = []
        for item in tqdm(jsonlist):
            questions = item["questions"]
            context = item["context"]
            answer = item["answer"]

            predictions = []

            document = StringIterableReader().load_data(texts=[context])
            parser = SimpleNodeParser.from_defaults(chunk_size=300, chunk_overlap=20)
            nodes = parser.get_nodes_from_documents(document)
            index = TreeIndex(nodes)
            query_engine = index.as_query_engine(model_name=model_name)

            for question in questions:
                prediction = query_engine.query("Answer the question with the letters of the correct options (e.g., A, B, C, or D) without including text. "+question).response
                predictions.append(prediction)

            output_item = {
                "context": context,
                "questions": questions,
                "num_question": len(questions),
                "answer": answer,
                "prediction": predictions,
                "dataset": file.split(".")[0],
            }
            output_list.append(output_item)

        output_file = '../../outputs/llamaindex/gpt-4o/' + file
        with jsonlines.open(output_file, mode='w') as writer:
            writer.write_all(output_list)
        print("Finished processing: " + file)        
