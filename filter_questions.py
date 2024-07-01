import asyncio
import aiohttp
import os
import json
import jsonlines
from tqdm import tqdm

api_key = "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"

async def fetch_response(session, api_key, messages):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-2024-05-13",
        "messages": messages,
        "temperature":0
    }
    
    async with session.post(url, headers=headers, json=data) as response:
        response_data = await response.json()
        return response_data

async def main():

    # Path to the folder containing the datasets
    folder_path = 'datasets/full_QA'

    # List of dataset filenames
    dataset_files = [
        'naturalquestion.jsonl',
    ]

    for file in dataset_files:
        print("Start to process: "+file)
        with jsonlines.open(folder_path+"/"+file) as f:
            jsonlist = list(f)

        question_list = []
        for item in jsonlist:
            for question in item["questions"]:
                question_list.append(question)

        batch_size = 5
        for i in tqdm(range(0, len(question_list), batch_size)):
            batch = question_list[i:i+batch_size]
            if len(batch) == 0:
                continue
            messages_list = [
                [{"role": "user", "content": "Answer briefly with no explanation, "+qa}] for qa in batch
            ]
        
            async with aiohttp.ClientSession() as session:
                tasks = [fetch_response(session, api_key, messages) for messages in messages_list]
                responses = await asyncio.gather(*tasks)
                
                for i in range(len(responses)):
                    question = batch[i]
                    output = responses[i]['choices'][0]['message']['content']
                    with open('outputs/unfiltered_full_set/'+file, 'a') as outfile:
                        json.dump([question,output], outfile)
                        outfile.write('\n')

# Run the main function
asyncio.run(main())
