# **Comprehensive Evaluation Tool for Text Generation**

## **Overview**

This project provides a comprehensive evaluation tool for text generation models. It supports both **online** and **offline** evaluations:

### **Online Evaluation**
The online evaluation leverages a Large Language Model (LLM) to assess the quality of generated responses based on several metrics, including:
- **Comprehensiveness**: How well the response covers the required information.
- **Groundedness**: How well the response is grounded in the source content.
- **Relevance**: How relevant the response is to the given context or query.

### **Offline Evaluation**
The offline evaluation calculates various traditional metrics to assess the performance of text generation models, such as:
- **ROUGE**: A set of metrics for evaluating automatic summarization and machine translation.
- **BLEU**: A metric for comparing a candidate text to one or more reference texts.
- **METEOR**: A metric that includes synonyms, stemming, and paraphrase matching.
- **BERTScore**: A metric that uses BERT embeddings to measure the similarity between sentences.
- **F1-Score**: A metric that balances precision and recall in evaluating model outputs.

## **Prerequisites**

Before using this tool, ensure you have the following installed:
- Python 3.8 or higher
- Necessary Python libraries (`pip install -r requirements.txt`)
- Access to an API key for the online evaluation (if applicable)

## **Setup**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
