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

This repository includes a bash script (`run.sh`) that facilitates running both online and offline evaluations on a dataset provided in a CSV file.

### Prerequisites

Ensure that you have the necessary Python environment set up with the required dependencies. You can install them using:

```bash
pip install -r requirements.txt
```

### Running the Evaluation Script
To run the evaluation script, follow the steps below:

1 - Provide the path to your input CSV file:

The script will prompt you to enter the path to the CSV file containing the dataset you wish to evaluate.

Choose Online Evaluation (Optional):

You will be asked if you want to perform an online evaluation. If you choose "yes," you will need to provide the following details:

Source column name: The column in the CSV file containing the source text.
Target column name: The column in the CSV file containing the target or reference text.
Metric: The evaluation metric you want to use (options include "comprehensiveness," "groundedness," or "relevance").
API key: Your API key for accessing the external LLM service.
Save format: The format in which you want to save the results ("csv" or "json").
Save path: The path where the results should be saved (excluding the file extension).
Choose Offline Evaluation (Optional):

You will also be asked if you want to perform an offline evaluation. If you choose "yes," you will need to provide the following details:

Log file path: The path where the offline evaluation log should be saved, including the file extension (e.g., .log).
Label column: The column in the CSV file containing the reference or ground truth text.
Prediction column: The column in the CSV file containing the generated text to be evaluated.
Metrics: The list of metrics to evaluate (e.g., ExactMatch, F1Score, BLEUScore).
Model name: The model name for the tokenizer and metrics. If you don't specify a model, a default model (unsloth/Meta-Llama-3.1-8B-Instruct) will be used.
Run the Evaluations:

The script will run the selected evaluations based on your input. If neither online nor offline evaluation is selected, the script will notify you that no evaluation was chosen.

Example Usage:

Here's an example of how the script might be used:
```
bash
./run.sh
```
Follow the prompts to provide the necessary inputs. The script will execute the evaluations and save the results as specified.
