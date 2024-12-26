# Long Context vs. RAG for LLMs: An Evaluation and Revisits

## Preparation

### Environment
To run our code, please install all the dependency packages by using the following command:

```
pip install -r requirements.txt
```

### Dataset
Download Dataset from the link：
[Download Dataset](https://entuedu-my.sharepoint.com/:f:/g/personal/xinze002_e_ntu_edu_sg/Elj-wpX68UpLvMMuvGmTn_UBDeepoSYfKEtukRUhGWJ5kw?e=sBQWeD)

Please make sure your data folder structure as below.
```bash
Datasets
  ├── full_set
  │   ├── 2wikimultihop.jsonl
  │   ├── coursera.jsonl
  │   └── (more datasets ... )
  │
  ├── full_set_filtered
  │   ├── 2wikimultihop.jsonl
  │   ├── coursera.jsonl
  │   └── (more datasets ... )
  │  
  ├── sample_set
  │   ├── 2wikimultihop.jsonl
  │   ├── coursera.jsonl
  │   └── (more datasets ... )
  │   
  ├── sample_set_filtered
  │   ├── 2wikimultihop.jsonl
  │   ├── coursera.jsonl
  │   └── (more datasets ... )
