# Long Context vs. RAG for LLMs: An Evaluation and Revisits
Extending context windows (i.e., Long Context, LC) and using retrievers to selectively access relevant information (i.e., RetrievalAugmentedGeneration, RAG)arethetwomain strategies to enable LLMs to incorporate extremely long external contexts. This paper revisits recent studies on this topic, highlighting their key insights and discrepancies. We then provide a more comprehensive evaluation by filtering out questions answerable without external context, identifying the most effective retrieval methods, and expanding the datasets. We show that LC generally outperforms RAG in question-answering benchmarks, especially for Wikipedia-based questions. Summarization-based retrieval performs comparably to LC, while chunk-based retrieval lags behind. However, RAG has advantages in dialogue-based and general question queries. These insights underscore the trade-offs between RAG and LC strategies, offering guidance for future optimization of LLMs with external knowledge sources. We also provide an in-depth discussion on this topic, highlighting the overlooked importance of context relevance in existing studies.

## Paper Link
[Download Paper](https://arxiv.org/pdf/2501.01880)

## Graph

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

## Citation
Please cite our paper if you find it helpful to your work:
```bibtex
@misc{li2024longcontextvsrag,
      title={Long Context vs. RAG for LLMs: An Evaluation and Revisits}, 
      author={Xinze Li and Yixin Cao and Yubo Ma and Aixin Sun},
      year={2024},
      eprint={2501.01880},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.01880}, 
}
