import os
import pandas as pd
import logging
import string
import numpy as np
from collections import Counter
from abc import ABC, abstractmethod
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer , AutoModel
from rouge_score import rouge_scorer
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
        pred_tokens = self.tokenizer.tokenize(str(prediction))
        ref_tokens = self.tokenizer.tokenize(str(reference))
        
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
        pred_tokens = self.tokenizer.tokenize(str(prediction))
        ref_tokens = self.tokenizer.tokenize(str(reference))
        return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.smoothie)

class ROUGEScore(Metric):
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

    def compute(self, prediction: str, reference: str) -> dict:
        scores = self.scorer.score(reference, prediction)
        return {
            'rouge-1': scores['rouge1'].fmeasure,
            'rouge-2': scores['rouge2'].fmeasure,
            'rouge-l': scores['rougeL'].fmeasure
        }

class METEORScore(Metric):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute(self, prediction: str, reference: str) -> float:
        prediction = self.tokenizer.tokenize(prediction)
        reference = self.tokenizer.tokenize(reference)
        return meteor_score([reference], prediction)


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
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
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


# # Example usage
# if __name__ == "__main__":
#     df = pd.read_csv("data/TEST/mrc_llama405.csv")
#     pred = pd.read_csv("loggers/lora32-response-to-llama405-mrc.csv")
#     tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
#     metrics = [ExactMatch(), F1Score(tokenizer),BERTScore(device='cuda'),BLEUScore(tokenizer),ROUGEScore(),METEORScore(tokenizer),]
#     evaluator = Evaluator(task="summarization", metrics=metrics)
#     references = df['answer'].tolist()
#     predictions = pred['assistant'].tolist()
#     evaluator.evaluate(predictions, references)