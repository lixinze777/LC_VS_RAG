#!/bin/bash

set -e
python rag/contriever/answer_questions_with_contriever.py
python rag/contriever/answer_questions_with_contriever_mcq.py

echo "Contriever scripts have been executed successfully."
