# Evaluation Script for Online and Offline Metrics

## Introduction

This repository provides a comprehensive evaluation tool for assessing the quality of generated responses using both **online** and **offline** evaluation methods.

### Online Evaluation

Online evaluation leverages a Language Learning Model (LLM) to evaluate the quality of generated responses based on various metrics. These metrics include:

- **Comprehensiveness**: Assesses whether the response covers all key points and details provided in the input.
- **Groundedness**: Measures how well the response is supported by the provided source information.
- **Relevance**: Determines the relevance of the response to the given prompt.

These evaluations are performed using an external LLM service, and the results can be saved in either CSV or JSON format.

### Offline Evaluation

Offline evaluation involves the use of traditional NLP metrics to assess the quality of generated responses. These metrics do not rely on an external service and include:

- **ROUGE**: Measures the overlap of n-grams between the generated text and the reference.
- **BLEU**: Evaluates how many n-grams in the generated text match the reference text.
- **METEOR**: Considers precision, recall, and synonymy for evaluating text generation quality.
- **BERTScore**: Uses contextual embeddings from BERT to compare the similarity of generated text and reference text.
- **F1 Score**: A harmonic mean of precision and recall, typically used for exact match evaluation.
- **ExactMatch**: Checks if the generated text exactly matches the reference text, ignoring case and leading/trailing spaces, returning 1.0 for a match and 0.0 otherwise

## How to Use the Bash Script

This repository includes a bash script (`run_evaluation.sh`) that facilitates running both online and offline evaluations on a dataset provided in a CSV file.

### Prerequisites

Ensure that you have the necessary Python environment set up with the required dependencies. You can install them using:

```bash
pip install -r requirements.txt
