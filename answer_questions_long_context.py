import asyncio
import aiohttp
import os
import json
import jsonlines
from tqdm import tqdm

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
        "temperature":0
    }
    
    async with session.post(url, headers=headers, json=data) as response:
        response_data = await response.json()
        return response_data

async def main():

    # Path to the folder containing the datasets
    folder_path = 'datasets/full_filtered_QA'

    # List of dataset filenames

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
    ]

    for file in dataset_files:
        print("Start to process: "+file)
        with jsonlines.open(folder_path+"/"+file) as f:
            jsonlist = list(f)

        find_original_context = {}
        question_list = []
        for item in jsonlist:
            questions = item["questions"]
            context = item["context"]

            max_token_length = 360000
            if len(context) > max_token_length:
                context = context[:max_token_length]
            q = "From the context: "+context+", answer the questions briefly with no explanation. If there are multiple questions, split answers with [sep]."
            for question in questions:
                q = q + question +"[sep]"
            q = q[:-5]
            question_list.append(q)
            find_original_context[q] = item

        batch_size = 5
        for i in tqdm(range(0, len(question_list), batch_size)):
            batch = question_list[i:i+batch_size]
            if len(batch) == 0:
                continue
            messages_list = [
                [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020? [sep] Which country won world cup 2022?"},
                {"role": "assistant", "content": "Los Angeles Dodgers [sep] Argentina"},
                {"role": "user", "content": qa}] for qa in batch
            ]
        
            async with aiohttp.ClientSession() as session:
                tasks = [fetch_response(session, api_key, messages) for messages in messages_list]
                responses = await asyncio.gather(*tasks)
                
                for i in range(len(responses)):
                    question = batch[i]
                    try:
                        output = responses[i]['choices'][0]['message']['content']
                    except Exception as e:
                        print(responses)
                        raise SystemExit(e)
                    with open('outputs/long_full_set/gpt-4o/'+file, 'a') as outfile:
                        item = find_original_context[question]
                        item["prediction"] = output
                        json.dump(item, outfile)
                        outfile.write('\n')

# Run the main function
asyncio.run(main())
