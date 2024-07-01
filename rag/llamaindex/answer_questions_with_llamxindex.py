import asyncio
import aiohttp
import os
import json
import jsonlines
from tqdm import tqdm
from llama_index.core import GPTSimpleVectorIndex, Document
from llama_index.core.readers.schema.base import DocumentSchema

api_key = "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"
model_name = "gpt-4o-2024-05-13"

async def fetch_response(session, api_key, messages):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": 0
    }
    
    async with session.post(url, headers=headers, json=data) as response:
        response_data = await response.json()
        return response_data

def retrieve_relevant_chunks_llama_index(context, question, num_chunks=5):
    documents = [DocumentSchema(text=doc) for doc in context]
    index = GPTSimpleVectorIndex(documents)
    
    query_response = index.query(question)
    relevant_chunks = [doc.text for doc in query_response[:num_chunks]]
    
    return relevant_chunks

async def main():
    folder_path = '../../datasets/filtered_QA'
    dataset_files = [
        'coursera.jsonl',
        '2wikimultihopqa.jsonl',
        'hotpotqa.jsonl',
        'multifieldqa.jsonl',
        'naturalquestion.jsonl',
        'narrativeqa.jsonl',
        'multidoc2dial.jsonl',
        'qasper.jsonl',
        'quality.jsonl',
        'toeflqa.jsonl',
        'musique.jsonl',
        'novelqa.jsonl'
    ]
    dataset_files = [
        'naturalquestion.jsonl',
        '2wikimultihopqa.jsonl',
        'hotpotqa.jsonl',
        'multifieldqa.jsonl',
        'narrativeqa.jsonl',
        'multidoc2dial.jsonl',
        'qasper.jsonl',
        'musique.jsonl',
    ]

    async with aiohttp.ClientSession() as session:
        for file in dataset_files:
            print("Start to process: " + file)
            with jsonlines.open(folder_path + "/" + file) as f:
                jsonlist = list(f)
            
            output_list = []
            for item in tqdm(jsonlist):
                questions = item["questions"]
                context = item["context"]
                answer = item["answer"]

                predictions = []
                for question in questions:
                    # Retrieve relevant chunks for each question using llama-index
                    chunks = retrieve_relevant_chunks_llama_index(context, question)
                    combined_chunks = " ".join(chunks)
                    
                    q = "From the context: " + combined_chunks + ", answer the question briefly with no explanation. " + question
                    messages = [{"role": "user", "content": q}]
                    response_data = await fetch_response(session, api_key, messages)
                    prediction = response_data['choices'][0]['message']['content']
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

            output_file = '../../outputs/llama_index/gpt-4o/' + file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with jsonlines.open(output_file, mode='w') as writer:
                writer.write_all(output_list)
            print("Finished processing: " + file)

if __name__ == "__main__":
    asyncio.run(main())
