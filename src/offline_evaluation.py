import os
import pandas as pd
import logging
import string
import numpy as np
import argparse
from collections import Counter
from abc import ABC, abstractmethod
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer , AutoModel
from nltk.translate.meteor_score import meteor_score

import warnings
warnings.filterwarnings("ignore")

class Preprocessor:
    def __init__(self):
        pass

    def _remove_punctuation(self, text):
        """
        Removes English and Persian punctuation from the input text.
        """
        persian_punctuation = '،؛؟'
        translator = str.maketrans('', '', string.punctuation + persian_punctuation)
        return text.translate(translator)

    def _persian_to_english_number(self, persian_num):
        """
        Converts Persian-style numbers in a string to English numbers.
        """
        persian_to_english_map = {
            '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
            '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
        }
        english_num = ''.join(persian_to_english_map.get(char, char) for char in persian_num)
        return english_num

    def _convert_non_answers(self, target):
        actual_out = "پاسخ سوال در متن یافت نشد"
        suggested_out = "این اطلاعات در متن موجود نیست."
        return target.replace(actual_out, suggested_out)

    def __call__(self, text):
        text = self._convert_non_answers(text)
        text = self._persian_to_english_number(text)
        text = self._remove_punctuation(text)
        return text



class Metric(ABC):
    @abstractmethod
    def compute(self, prediction: str, reference: str) -> float:
        pass

class ExactMatch(Metric):
    def compute(self, prediction: str, reference: str) -> float:
        return float(str(prediction).strip().lower() == str(reference).strip().lower())

class F1Score(Metric):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute(self, prediction: str, reference: str) -> float:
        if self.tokenizer:
            pred_tokens = self.tokenizer.tokenize(prediction)
            ref_tokens = self.tokenizer.tokenize(reference)
        else:
            pred_tokens = prediction.split()
            ref_tokens = reference.split()
        
        common_tokens = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common_tokens.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        return 2 * (precision * recall) / (precision + recall)

class BLEUScore(Metric):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.smoothie = SmoothingFunction().method4

    def compute(self, prediction: str, reference: str) -> float:
        if self.tokenizer:
            pred_tokens = self.tokenizer.tokenize(prediction)
            ref_tokens = self.tokenizer.tokenize(reference)
        else:
            pred_tokens = prediction.split()
            ref_tokens = reference.split()

        return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.smoothie)


class ROUGEScore(Metric):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def compute(self, prediction: str, reference: str) -> dict:
        if not prediction.strip() or not reference.strip():  # Check if either string is empty
            return {
                'rouge-1': 0.0,
                'rouge-2': 0.0,
                'rouge-l': 0.0
            }
        
        rouge_1_score = self._rouge_n(prediction, reference, n=1)
        rouge_2_score = self._rouge_n(prediction, reference, n=2)
        rouge_l_score = self._rouge_l(prediction, reference)
        
        return {
            'rouge-1': rouge_1_score,
            'rouge-2': rouge_2_score,
            'rouge-l': rouge_l_score
        }

    def _n_grams(self, sequence, n):
        return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]

    def _lcs_length(self, x, y):
        m = len(x)
        n = len(y)
        table = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    table[i][j] = table[i - 1][j - 1] + 1
                else:
                    table[i][j] = max(table[i][j - 1], table[i - 1][j])
        return table[m][n]

    def _rouge_n(self, prediction, reference, n):
        if self.tokenizer:
            pred_tokens = self.tokenizer.tokenize(prediction)
            ref_tokens = self.tokenizer.tokenize(reference)
        else:
            pred_tokens = prediction.split()
            ref_tokens = reference.split()
        
        pred_ngrams = Counter(self._n_grams(pred_tokens, n))
        ref_ngrams = Counter(self._n_grams(ref_tokens, n))
        
        overlap_ngrams = pred_ngrams & ref_ngrams  # Intersection: min(count in pred, count in ref)
        overlap_count = sum(overlap_ngrams.values())
        
        if sum(ref_ngrams.values()) == 0:
            return 0.0
        
        rouge_n_recall = overlap_count / sum(ref_ngrams.values())  # Recall
        
        return rouge_n_recall

    def _rouge_l(self, prediction, reference):
        pred_tokens = self.tokenizer.tokenize(prediction)
        ref_tokens = self.tokenizer.tokenize(reference)
        
        lcs = self._lcs_length(pred_tokens, ref_tokens)
        
        if len(ref_tokens) == 0:
            return 0.0
        
        rouge_l_recall = lcs / len(ref_tokens)  # LCS-based recall
        
        return rouge_l_recall


class METEORScore(Metric):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute(self, prediction: str, reference: str) -> float:
        if self.tokenizer:
            pred_tokens = self.tokenizer.tokenize(prediction)
            ref_tokens = self.tokenizer.tokenize(reference)
        else:
            pred_tokens = prediction.split()
            ref_tokens = reference.split()
        return meteor_score([ref_tokens], pred_tokens)


class BERTScore(Metric):
    def __init__(self, model_name="PartAI/TookaBERT-Base",device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def _cosine_similarity(self, a, b):
        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _bert_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}  # Move each tensor to the device
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0).cpu().detach().numpy()  # Squeeze to remove batch dimension
        return embeddings

    def _precision_bert(self, pred_embeddings, ref_embeddings):
        precision_scores = []
        for pred_vector in pred_embeddings:
            max_similarity = max(self._cosine_similarity(pred_vector, ref_vector) for ref_vector in ref_embeddings)
            precision_scores.append(max_similarity)
        return np.mean(precision_scores)

    def _recall_bert(self, pred_embeddings, ref_embeddings):
        recall_scores = []
        for ref_vector in ref_embeddings:
            max_similarity = max(self._cosine_similarity(ref_vector, pred_vector) for pred_vector in pred_embeddings)
            recall_scores.append(max_similarity)
        return np.mean(recall_scores)

    def _f1_score(self, precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def compute(self, prediction: str, reference: str) -> dict:
        embeddings1 = self._bert_embeddings(prediction)
        embeddings2 = self._bert_embeddings(reference)
        
        precision = self._precision_bert(embeddings1, embeddings2)
        recall = self._recall_bert(embeddings1, embeddings2)
        f1 = self._f1_score(precision, recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }




class Evaluator:
    def __init__(self, task: str, metrics: list, logfile=None, tokenizer=None):
        self.task = task
        self.metrics = metrics
        self.tokenizer = tokenizer
        self.logfile = logfile if logfile is not None else "./test.log"
        self.preprocessor = Preprocessor()
        self._setup_logger()


    def _preprocess(self,predictions,references):
        predictions = [self.preprocessor(prediction) for prediction in predictions]
        references = [self.preprocessor(reference) for reference in references]
        return predictions , references
    
    def _setup_logger(self):
        log_dir = os.path.dirname(self.logfile)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if os.path.exists(self.logfile):
            open(self.logfile, 'w').close()

        logging.basicConfig(filename=self.logfile, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def evaluate(self, predictions, references):
        results = {}
        num_examples = len(predictions)

        predictions , references = self._preprocess(predictions,references)
    
        # Initialize results based on the metric types
        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            if isinstance(metric.compute("", ""), dict):  # Check if the metric returns a dictionary
                results[metric_name] = {}
            else:
                results[metric_name] = 0
    
        for i, (pred, ref) in enumerate(tqdm(zip(predictions, references), total=num_examples)):
            self.logger.info(f'Example {i+1}:')
            self.logger.info(f'Prediction: {pred}')
            self.logger.info(f'Reference: {ref}')
            
            for metric in self.metrics:
                metric_name = metric.__class__.__name__
                score = metric.compute(pred, ref)
    
                if isinstance(score, dict):  # Handle dictionary return types
                    for sub_metric, value in score.items():
                        if sub_metric not in results[metric_name]:
                            results[metric_name][sub_metric] = 0
                        results[metric_name][sub_metric] += value
                    self.logger.info(f'{metric_name}: {score}')
                else:
                    results[metric_name] += score
                    self.logger.info(f'{metric_name}: {score:.4f}')
    
                self.logger.info('-' * 80)
    
        # Calculate and log average scores
        for metric_name, total_score in results.items():
            if isinstance(total_score, dict):
                for sub_metric, value in total_score.items():
                    avg_score = value / num_examples
                    self.logger.info(f'Average {metric_name} {sub_metric}: {avg_score:.4f}')
                    print(f'Average {metric_name} {sub_metric}: {avg_score:.4f}')
            else:
                avg_score = total_score / num_examples
                self.logger.info(f'Average {metric_name}: {avg_score:.4f}')
                print(f'Average {metric_name}: {avg_score:.4f}')


def get_metric_objects(metric_names, tokenizer=None, device="cpu"):
    """
    Create a list of metric objects based on the provided metric names.

    :param metric_names: List of metric names as strings.
    :param tokenizer: The tokenizer to use for certain metrics.
    :param device: The device to use for certain metrics.
    :return: A list of instantiated metric objects.
    """
    metrics_dict = {
        "exact_match": ExactMatch,
        "f1_score": lambda: F1Score(tokenizer),
        "bert_score": lambda: BERTScore(device=device),
        "bleu": lambda: BLEUScore(tokenizer),
        "rouge": lambda: ROUGEScore(tokenizer),
        "meteor": lambda: METEORScore(tokenizer),
    }

    return [metrics_dict[metric]() for metric in metric_names if metric in metrics_dict]

def main():
    parser = argparse.ArgumentParser(description="Perform offline evaluation using specified metrics.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file containing the dataset.")
    parser.add_argument('--metrics', type=str, nargs='+', required=True, help="List of metrics to evaluate.")
    parser.add_argument("--label_column", required=True, help="Name of the column containing the true labels.")
    parser.add_argument("--prediction_column", required=True, help="Name of the column containing the predictions.")
    parser.add_argument("--logfile", default="results/metrics.log", help="Path to save the evaluation log file.")
    parser.add_argument("--model_name",required=False,default=None, help="Model for tokenizer and metrics.")

    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)
    references = df[args.label_column].tolist()
    predictions = df[args.prediction_column].tolist()

    # Load tokenizer
    if args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = None

    # Initialize metrics
    metrics = get_metric_objects(metric_names=args.metrics,tokenizer=tokenizer)
    evaluator = Evaluator(task=None, metrics=metrics, logfile=args.logfile, tokenizer=tokenizer)
    evaluator.evaluate(predictions, references)

if __name__ == "__main__":
    main()