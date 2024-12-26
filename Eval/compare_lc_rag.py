import pandas as pd
import csv
import json
import re
import numpy as np

model_name = "gpt-4o" # gpt-4-turbo
rag_setting = "raptor_full_set" #"bm25" #raptor
long_path = "../long_full_set/" + model_name +"/"
rag_path = "../"+rag_setting+"/"  + model_name +"/"
json_path = "../../datasets/full_set_filtered/"

def write_list_to_csv(filename, data_list):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["question", "context_word_count", "source"])
        for item in data_list:
            writer.writerow(item)

def get_context_from_json(json_file, question):
    with open(json_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            if question in data["questions"]:
                return data["context"]
    return ""

def categorize_question(question):
    question = question.lower()
    if "how" in question:
        return "How"
    elif "why" in question:
        return "Why"
    elif "what" in question:
        return "What"
    elif "when" in question:
        return "When"
    elif "where" in question:
        return "Where"
    elif "which" in question:
        return "Which"
    elif "who" in question:
        return "Who"
    else:
        return "Others"

def calculate_stats(data):
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std_dev': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        '25th_percentile': np.percentile(data, 25),
        '75th_percentile': np.percentile(data, 75)
    }

dataset_files = {
    'coursera.csv': 'course website',
    '2wikimultihopqa.csv': 'Wikipedia',
    'hotpotqa.csv': 'Wikipedia',
    'multifieldqa.csv': 'paper or report',
    'naturalquestion.csv': 'Wikipedia',
    'narrativeqa.csv': 'story',
    'qasper.csv': 'paper or report',
    'quality.csv': 'story',
    'toeflqa.csv': 'dialogue',
    'musique.csv': 'Wikipedia',
    'multidoc2dial.csv': 'dialogue',
    'novelqa.csv':'story'
}

total_qn = 0
overall_long = 0
overall_rag = 0
overall_long_only = 0
overall_rag_only = 0
overall_long_loose = 0
overall_rag_loose = 0
long_only_context_word_count = []
rag_only_context_word_count = []
long_only_question_word_count = []
rag_only_question_word_count = []
long_only_question_categories = {"How": 0, "Why": 0, "What": 0, "When": 0, "Where": 0, "Which": 0, "Who": 0, "Others": 0}
rag_only_question_categories = {"How": 0, "Why": 0, "What": 0, "When": 0, "Where": 0, "Which": 0, "Who": 0, "Others": 0}
source_stats = {source: {"long": 0, "rag": 0} for source in set(dataset_files.values())}
stats_by_file = {}
long_correct_only_questions = []
rag_correct_only_questions = []

for file, source in dataset_files.items():
    df_long = pd.read_csv(long_path+file)
    df_rag = pd.read_csv(rag_path+file)
    json_file = json_path + file[:-4] + ".jsonl"

    questions = df_long["question"].tolist()
    long_label = df_long["label"].tolist()
    rag_label = df_rag["label"].tolist()

    num_long_correct = long_label.count(1)
    num_rag_correct = rag_label.count(1)

    long_correct_only = [index for index, (l, r) in enumerate(zip(long_label, rag_label)) if l == 1 and r == 0]
    rag_correct_only = [index for index, (l, r) in enumerate(zip(long_label, rag_label)) if r == 1 and l == 0]
    both_wrong = [index for index, (l, r) in enumerate(zip(long_label, rag_label)) if l == 0 and r == 0]

    total_qn += len(long_label)

    for i in long_correct_only:
        context = get_context_from_json(json_file, questions[i])
        long_only_context_word_count.append(len(context.split()))
        long_only_question_word_count.append(len(questions[i].split()))
        long_correct_only_questions.append((questions[i], len(context.split()), source))
        category = categorize_question(questions[i])
        long_only_question_categories[category] += 1
        source_stats[source]["long"] += 1

    for i in rag_correct_only:
        context = get_context_from_json(json_file, questions[i])
        rag_only_context_word_count.append(len(context.split()))
        rag_only_question_word_count.append(len(questions[i].split()))
        rag_correct_only_questions.append((questions[i], len(context.split()), source))
        category = categorize_question(questions[i])
        if category == "Others":
            print(questions[i])
        rag_only_question_categories[category] += 1
        source_stats[source]["rag"] += 1

    overall_long += num_long_correct
    overall_rag += num_rag_correct
    overall_long_only += len(long_correct_only)
    overall_rag_only += len(rag_correct_only)

    f1_long_only = 0
    f1_rag_only = 0
    if "f1" in list(df_long.keys()):
        f1_long = df_long["f1"].tolist()
        f1_rag = df_rag["f1"].tolist()
        for index in both_wrong:
            if f1_long[index] > f1_rag[index]:
                f1_long_only += 1
            elif f1_rag[index] > f1_long[index]:
                f1_rag_only += 1
    
    long_correct_only_loose = len(long_correct_only)+f1_long_only
    rag_correct_only_loose = len(rag_correct_only)+f1_rag_only
    overall_long_loose += long_correct_only_loose
    overall_rag_loose += rag_correct_only_loose

    stats_by_file[file[:-4]] = [len(long_label), num_long_correct, num_rag_correct, len(long_correct_only), len(rag_correct_only),long_correct_only_loose,rag_correct_only_loose]

stats_by_file["overall"] = [total_qn, overall_long, overall_rag, overall_long_only, overall_rag_only, overall_long_loose, overall_rag_loose]

columns = ["dataset", "num question", "long correct", "rag correct", "long correct only", "rag correct only", "long better", "rag better"]
df_stats = pd.DataFrame.from_dict(stats_by_file, orient='index', columns=columns[1:])
df_stats.reset_index(inplace=True)
df_stats.rename(columns={'index': 'dataset'}, inplace=True)

dir = "../statistics/"+rag_setting+"/"+model_name+"/"
output_csv_path = "data_stats.csv"
df_stats.to_csv(dir+output_csv_path, index=False)

write_list_to_csv(dir+'long_correct_only_questions.csv', long_correct_only_questions)
write_list_to_csv(dir+'rag_correct_only_questions.csv', rag_correct_only_questions)

long_context_stats = calculate_stats(long_only_context_word_count)
rag_context_stats = calculate_stats(rag_only_context_word_count)
long_question_stats = calculate_stats(long_only_question_word_count)
rag_question_stats = calculate_stats(rag_only_question_word_count)

with open(f"{dir}{model_name}_stats.txt", 'w') as f:
    f.write(f"Model: {model_name}\n")
    
    f.write("Long only context stats:\n")
    for stat, value in long_context_stats.items():
        f.write(f"{stat.capitalize()}: {value}\n")
    
    f.write("\nRag only context stats:\n")
    for stat, value in rag_context_stats.items():
        f.write(f"{stat.capitalize()}: {value}\n")
    
    f.write("\nLong only question stats:\n")
    for stat, value in long_question_stats.items():
        f.write(f"{stat.capitalize()}: {value}\n")
    
    f.write("\nRag only question stats:\n")
    for stat, value in rag_question_stats.items():
        f.write(f"{stat.capitalize()}: {value}\n")

    f.write("\nLong only questions:\n")
    for category, count in long_only_question_categories.items():
        percentage = (count / overall_long_only) * 100 if overall_long_only > 0 else 0
        f.write(f"Category '{category}': {count} questions, {percentage:.2f}%\n")
    
    f.write("\nRag only questions:\n")
    for category, count in rag_only_question_categories.items():
        percentage = (count / overall_rag_only) * 100 if overall_rag_only > 0 else 0
        f.write(f"Category '{category}': {count} questions, {percentage:.2f}%\n")
    
    f.write("\nSource statistics:\n")
    for source, counts in source_stats.items():
        long_percentage = (counts["long"] / overall_long_only) * 100 if overall_long_only > 0 else 0
        rag_percentage = (counts["rag"] / overall_rag_only) * 100 if overall_rag_only > 0 else 0
        f.write(f"Source '{source}': {counts['long']} long questions ({long_percentage:.2f}%), {counts['rag']} rag questions ({rag_percentage:.2f}%)\n")

print(f"Model: {model_name}")

print("Long only context stats:")
for stat, value in long_context_stats.items():
    print(f"{stat.capitalize()}: {value}")
    
print("\nRag only context stats:")
for stat, value in rag_context_stats.items():
    print(f"{stat.capitalize()}: {value}")

print("\nLong only question stats:")
for stat, value in long_question_stats.items():
    print(f"{stat.capitalize()}: {value}")

print("\nRag only question stats:")
for stat, value in rag_question_stats.items():
    print(f"{stat.capitalize()}: {value}")

print("\nLong only questions:")
for category, count in long_only_question_categories.items():
    percentage = (count / overall_long_only) * 100 if overall_long_only > 0 else 0
    print(f"Category '{category}': {count} questions, {percentage:.2f}%")
    
print("\nRag only questions:")
for category, count in rag_only_question_categories.items():
    percentage = (count / overall_rag_only) * 100 if overall_rag_only > 0 else 0
    print(f"Category '{category}': {count} questions, {percentage:.2f}%")

print("\nSource statistics:")
for source, counts in source_stats.items():
    long_percentage = (counts["long"] / overall_long_only) * 100 if overall_long_only > 0 else 0
    rag_percentage = (counts["rag"] / overall_rag_only) * 100 if overall_rag_only > 0 else 0
    print(f"Source '{source}': {counts['long']} long questions ({long_percentage:.2f}%), {counts['rag']} rag questions ({rag_percentage:.2f}%)")