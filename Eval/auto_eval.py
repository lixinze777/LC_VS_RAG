import jsonlines
import csv
import re
import string

model_name = "gpt-4o/" #"gpt-4o/"#"gpt-4o-2024-05-13/" #"gpt-4-turbo/"
setting = "raptor_full_set/" #"openai/" #"bm25/" #"raptor/"#"long/" #"bm25/" #"long_full_set/"
prediction_path = "../../outputs/"


dataset_files = [
    'coursera.jsonl',
    '2wikimultihopqa.jsonl',
    'hotpotqa.jsonl',
    'multifieldqa.jsonl',
    'naturalquestion.jsonl',
    'narrativeqa.jsonl',
    'qasper.jsonl',
    'quality.jsonl',
    'toeflqa.jsonl',
    'musique.jsonl',
    'novelqa.jsonl',
    'multidoc2dial.jsonl'
]


def normalize_mcq_answer(input_string):
    input_string = re.sub(r'Question\s*\d+\.?\s*', '', input_string)

    if '?' in input_string:
        parts = input_string.rsplit('?', 1)
        after_question = parts[1].strip()
        result = re.split(r'\.', after_question)[0].strip()
    else:
        result = input_string.strip()
    result = result.replace(".","").replace(",","").replace(" ","").strip()

    if len(result) > 4:
        result = result[0]
    return result

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the|and|or|about|to)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))

def EM(s1,s2):
    list1 = s1.split()
    list2 = s2.split()

    return all(item in list2 for item in list1) or all(item in list1 for item in list2)

def f1_score(prediction, gold):
    prediction_set = set(prediction.split())
    gold_set = set(gold.split())
    true_positives = len(prediction_set & gold_set)
    false_positives = len(prediction_set - gold_set)
    false_negatives = len(gold_set - prediction_set)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1,2)


def split_prediction(file, prediction, num_question):
    predictions = prediction.split("[sep]")

    while len(predictions) < num_question:
        predictions.append("NA")
    try:
        assert len(predictions) == num_question
    except AssertionError as e:
        print("error file is: "+file)
        print("num question should be: "+str(num_question))
        print(prediction)
    return predictions

def process_novelqa():
    # Please Obtain Evaluation Results from https://www.codabench.org/competitions/2727/#/participate-tab
    pass

def process_mcq(file):
    with jsonlines.open(prediction_path+setting+model_name+file) as f:
        jsonlist = list(f)
    all_question = []
    all_answer = []
    all_prediction = []
    all_label = []
    for item in jsonlist:
        questions = item["questions"]
        num_question = item["num_question"]
        answers = item["answer"]
        predictions = item["prediction"]

        if isinstance(predictions,str):
            if num_question == 1:
                predictions = [predictions]
            else:
                predictions = split_prediction(file, predictions, num_question)

        for i in range(num_question):
            question = questions[i]
            answer = answers[i]
            prediction = normalize_mcq_answer(predictions[i])
            label = 0
            if answer == prediction:
                label = 1
            all_question.append(question)
            all_answer.append(answer)
            all_prediction.append(prediction)
            all_label.append(label)
    with open("../"+setting+model_name+file[:-6]+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["question", "answer", "prediction","label"])
        # Write the rows
        for question, answer, prediction, label in zip(all_question, all_answer, all_prediction, all_label):
            writer.writerow([question, answer, prediction, label])

def process_open(file):
    with jsonlines.open(prediction_path+setting+model_name+file) as f:
        jsonlist = list(f)
    all_question = []
    all_answer = []
    all_prediction = []
    all_label = []
    all_f1 = []
    for item in jsonlist:
        questions = item["questions"]
        num_question = item["num_question"]
        answers = item["answer"]
        predictions = item["prediction"]

        if isinstance(predictions,str):
            if num_question == 1:
                predictions = [predictions]
            else:
                predictions = split_prediction(file, predictions, num_question)

        for i in range(num_question):
            question = questions[i]
            answer = answers[i]
            prediction = predictions[i]
            label = 0
            _answer = normalize_answer(answer)
            _prediction = normalize_answer(prediction)
            if EM(_answer, _prediction):
                label = 1
            f1 = f1_score(_prediction, _answer)
            all_question.append(question)
            all_answer.append(answer)
            all_prediction.append(prediction)
            all_label.append(label)
            all_f1.append(f1)
            
    with open("../"+setting+model_name+file[:-6]+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["question", "answer", "prediction","label","f1"])
        # Write the rows
        for question, answer, prediction, label, f1 in zip(all_question, all_answer, all_prediction, all_label, all_f1):
            writer.writerow([question, answer, prediction, label, f1])
            
for file in dataset_files:
    if file == 'novelqa.jsonl':
        process_novelqa()
    elif file in ['coursera.jsonl','quality.jsonl','toeflqa.jsonl']:
        process_mcq(file)
    else:
        process_open(file)
