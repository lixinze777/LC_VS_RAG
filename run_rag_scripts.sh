#!/bin/bash
# change to corresponding RAG folder 
set -e
python RAG/contriever/answer_questions_with_contriever.py
python RAG/contriever/answer_questions_with_contriever_mcq.py

echo "Contriever scripts have been executed successfully."
