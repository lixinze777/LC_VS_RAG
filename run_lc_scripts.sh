#!/bin/bash

set -e
python LC/answer_questions_with_long_context.py
python LC/answer_questions_with_long_context_mcq.py

echo "LC scripts have been executed successfully."
