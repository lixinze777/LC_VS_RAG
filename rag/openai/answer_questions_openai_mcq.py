import asyncio
import aiohttp
import os
import json
import jsonlines
from tqdm import tqdm
import openai
import numpy as np
import re

api_key = "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"
model_name = "gpt-4o-2024-05-13"
openai.api_key = "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"

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

def get_embedding(text, model="text-embedding-3-small"):
    response = openai.Embedding.create(input=text, model=model)
    embedding = response['data'][0]['embedding']
    return embedding


def get_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def chunk_text(context, chunk_size):
    sentences = re.split(r'(?<=[.!?]) +', context)
    
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def retrieve_relevant_chunks_for_question(context, questions, chunk_size, num_chunks):
    context_embeddings = [get_embedding(doc) for doc in chunk_text(context, chunk_size=chunk_size)]
    question_embeddings = [get_embedding(question) for question in questions]
    
    relevant_chunks = []
    for question_embedding in question_embeddings:
        similarities = [get_cosine_similarity(question_embedding, context_embedding) for context_embedding in context_embeddings]
        top_indices = np.argsort(similarities)[-num_chunks:]
        top_chunks = [context[idx] for idx in top_indices]
        relevant_chunks.extend(top_chunks)
    
    return relevant_chunks

async def main():
    folder_path = '../../datasets/filtered_QA'
    dataset_files = [
        'coursera.jsonl',
        'quality.jsonl',
        'toeflqa.jsonl',
        'novelqa.jsonl'
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
                    chunks = retrieve_relevant_chunks_for_question(context, question, chunk_size=300, num_chunks=5)
                    combined_chunks = " ".join(chunks)
                    
                    q = "From the context: " + combined_chunks + ", answer the question with the letters of the correct options (e.g., A, B, C, or D) without including text. " + question
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

            output_file = '../../outputs/openai/gpt-4o/' + file
            with jsonlines.open(output_file, mode='w') as writer:
                writer.write_all(output_list)
            print("Finished processing: " + file)

if __name__ == "__main__":
    asyncio.run(main())
