import os
from llama_index.core import Settings, VectorStoreIndex, TreeIndex
from llama_index.readers.string_iterable import StringIterableReader
from llama_index.core.node_parser import SimpleNodeParser, SentenceWindowNodeParser, SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
import logging
import sys
import jsonlines
from tqdm import tqdm
import numpy as np

api_key = "your key here"
os.environ["OPENAI_API_KEY"] = api_key
llm = OpenAI(model="gpt-4o-2024-05-13", temperature=0)
embed_name = "text-embedding-3-small"
text_splitter = SentenceSplitter()

Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(model = embed_name)
Settings.text_splitter = text_splitter

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
            node_parser = SentenceWindowNodeParser.from_defaults(
                        window_size=5,
                        window_metadata_key="window",
                        original_text_metadata_key="original_text",
                    )
            nodes = node_parser.get_nodes_from_documents(document)
            sentence_index = VectorStoreIndex(nodes)
            query_engine = sentence_index.as_query_engine(
                similarity_top_k=2,
                node_postprocessors=[
                    MetadataReplacementPostProcessor(target_metadata_key="window")
                ],
            )

            for question in questions:
                prediction = query_engine.query("answer the question with the letters of the correct options (e.g., A, BD, C, or ACD .etc) without including text."+question).response
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

        output_file = '../../outputs/windowindex/gpt-4o/' + file
        with jsonlines.open(output_file, mode='w') as writer:
            writer.write_all(output_list)
        print("Finished processing: " + file)        
