import os
import json
os.environ["OPENAI_API_KEY"] = "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"
from raptor import RetrievalAugmentation
from tqdm import tqdm
import jsonlines
import argparse

def answer_questions(tree, context, questions):

    # Initialize with default configuration. For advanced configurations, check the documentation. [WIP]
    RA = RetrievalAugmentation()
    SAVE_PATH = "trees/"+tree
    outputs = []

    if os.path.isfile(SAVE_PATH):
        RA = RetrievalAugmentation(tree=SAVE_PATH)
    else:
        RA.add_documents(context)
        RA.save(SAVE_PATH)

    #load questions:

    for question in questions:
        answer = RA.answer_question(question=question)
        outputs.append(answer)

    return outputs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default = '../datasets/filtered_QA')
    parser.add_argument("--output_path", type=str, default = '../outputs/raptor3/')
    parser.add_argument("--model_name", type=str, default="gpt-4o") #gpt-4-turbo #gpt-4o
    parser.add_argument("--start_point", type=int, default = 0)
    args = parser.parse_args()

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
        '2wikimultihopqa.jsonl',
        'hotpotqa.jsonl',
        'multifieldqa.jsonl',
        'narrativeqa.jsonl',
        'multidoc2dial.jsonl',
        'qasper.jsonl',
        'musique.jsonl',
        'naturalquestion.jsonl',
    ]

    for file in dataset_files:
        print("Start to process: "+file)
        with jsonlines.open(args.input_path+"/"+file) as f:
            jsonlist = list(f)
        
        file_name = file[:-6]
        tree_count = args.start_point
        for item in tqdm(jsonlist[args.start_point:]):
            tree_count += 1
            tree_name = file_name+"_"+str(tree_count)
            questions = item["questions"]
            context = item["context"]
            predictions = answer_questions(tree_name, context, questions)
            item["prediction"] = predictions
            with open(args.output_path+'/'+args.model_name+'/'+file, 'a') as outfile:
                json.dump(item, outfile)
                outfile.write('\n')
