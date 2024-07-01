import asyncio
import aiohttp
import os
import json
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

# Initialization of the Contriever model
model_name_contriever = "facebook/contriever"

tokenizer = AutoTokenizer.from_pretrained(model_name_contriever)
model = AutoModel.from_pretrained(model_name_contriever)

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

def retrieve_relevant_chunks_contriever(context, question, num_chunks=5):
    context_embeddings = [model(**tokenizer(doc, return_tensors='pt'))['last_hidden_state'].mean(dim=1) for doc in context]
    question_embedding = model(**tokenizer(question, return_tensors='pt'))['last_hidden_state'].mean(dim=1)
    
    scores = [torch.nn.functional.cosine_similarity(question_embedding, context_embedding).item() for context_embedding in context_embeddings]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_chunks]
    relevant_chunks = [context[i] for i in top_indices]
    
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
                    # Retrieve relevant chunks for each question using Contriever
                    chunks = retrieve_relevant_chunks_contriever(context, question)
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

            output_file = '../../outputs/contriever/gpt-4o/' + file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with jsonlines.open(output_file, mode='w') as writer:
                writer.write_all(output_list)
            print("Finished processing: " + file)

if __name__ == "__main__":
    asyncio.run(main())
