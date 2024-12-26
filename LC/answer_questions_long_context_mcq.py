import asyncio
import aiohttp
import os
import json
import jsonlines
from tqdm import tqdm

api_key = "your key here"
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

    folder_path = '../datasets/full_set_filtered'

    dataset_files = [
        'coursera.jsonl',
        'quality.jsonl',
        'toeflqa.jsonl',
        'novelqa.jsonl'
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

            # Concat the context if greater than the model limit
            max_token_length = 360000
            if len(context) > max_token_length:
                context = context[:max_token_length]
            q = "From the context: "+context+", Answer the question with the letters of the correct options (e.g., A, B, C, or D) without including text. If there are multiple questions, split answers with [sep]."
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
                {"role": "user", "content": "Which country won world cup 2022? A. France B. China C. Argentina D. Brazil [sep] Which football player is from Argentina? A. Messi B. Ronaldo C. Di Maria D. Maradona"},
                {"role": "assistant", "content": "C [sep] ACD"},
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
                    with open('../outputs/long_full_set/gpt-4o/'+file, 'a') as outfile:
                        item = find_original_context[question]
                        item["prediction"] = output
                        json.dump(item, outfile)
                        outfile.write('\n')

asyncio.run(main())
