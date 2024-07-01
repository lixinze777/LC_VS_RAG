import json

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def find_mismatched_lengths(data):
    mismatched = []
    for item in data:
        questions = item.get("questions", [])
        answers = item.get("answer", [])
        if len(questions) != len(answers):
            mismatched.append((questions, answers))
    return mismatched

def main():
    file_path = "naturalquestion.jsonl"
    data = load_jsonl(file_path)
    mismatched = find_mismatched_lengths(data)
    
    for questions, answers in mismatched:
        print("Questions:", questions)
        print("Answers:", answers)
        print("Lengths:", len(questions), len(answers))
        print("-----")

if __name__ == "__main__":
    main()
