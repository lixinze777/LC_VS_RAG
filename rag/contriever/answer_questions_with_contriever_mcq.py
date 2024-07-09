import os
import json
import jsonlines
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import torch
import re

# Initialization of the Contriever model
model_name_contriever = "facebook/contriever"
tokenizer = AutoTokenizer.from_pretrained(model_name_contriever)
model = AutoModel.from_pretrained(model_name_contriever)


api_key = "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"
model_name = "gpt-4o-2024-05-13"
client = OpenAI(api_key = api_key)

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


def retrieve_relevant_chunks_contriever(context, question, chunk_size, num_chunks):
    context_embeddings = [model(**tokenizer(doc, truncation=True, max_length=512, return_tensors='pt'))['last_hidden_state'].mean(dim=1) for doc in chunk_text(context, chunk_size)]
    question_embedding = model(**tokenizer(question, truncation=True, max_length=512, return_tensors='pt'))['last_hidden_state'].mean(dim=1)
    
    scores = [torch.nn.functional.cosine_similarity(question_embedding, context_embedding).item() for context_embedding in context_embeddings]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_chunks]
    relevant_chunks = [context[i] for i in top_indices]
    
    return relevant_chunks


def main():
    folder_path = '../../datasets/filtered_QA'
    dataset_files = [
        'coursera.jsonl',
        'quality.jsonl',
        'toeflqa.jsonl',
        'novelqa.jsonl'
    ]

    dataset_files = [
        'sample.jsonl',
    ]

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
                chunks = retrieve_relevant_chunks_contriever(context, question, chunk_size=300, num_chunks=5)
                combined_chunks = " ".join(chunks)
                
                q = "From the context: " + combined_chunks + ", answer the question with the letters of the correct options (e.g., A, B, C, or D) without including text. " + question
                response = client.chat.completions.create(
                    model=model_name,
                    temperature = 0,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": q}
                    ]
                )
                prediction = response.choices[0].message.content
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
    main()
