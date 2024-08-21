import argparse
import pandas as pd

from feedback_services import (
    MRCEvaluationService,
    SummarizationEvaluationService,
    QA_EvaluationService,
    DialogueEvaluationService,
)
from prompts import (
    MRCBasePrompt,
    PrecisionConcisenessEvaluator,
    ComprehensivenessPrompt,
)

class EvaluationFactory:
    def __init__(self, api_key):
        self.api_key = api_key

    def create_evaluator(self, task, evaluator_name):
        task_evaluator_map = {
            'mrc': {
                'PrecisionConcisenessEvaluator': lambda: MRCEvaluationService(api_key=self.api_key, evaluator=PrecisionConcisenessEvaluator()),
                # Add more evaluators for MRC if needed
            },
            'summarization': {
                'comprehensiveness': lambda: SummarizationEvaluationService(api_key=self.api_key, evaluator=ComprehensivenessPrompt()),

            },
            'qa': {
                'SomeQAEvaluator': lambda: QA_EvaluationService(api_key=self.api_key, evaluator=None),  # Replace None with actual evaluator
                # Add more evaluators for QA if needed
            }
        }

        if task not in task_evaluator_map:
            raise ValueError(f"Unknown task: {task}")

        if evaluator_name not in task_evaluator_map[task]:
            raise ValueError(f"Unknown evaluator: {evaluator_name} for task: {task}")

        return task_evaluator_map[task][evaluator_name]()


class EvaluationManager:
    def __init__(self, api_key):
        self.api_key = api_key
        self.factory = EvaluationFactory(api_key)

    def evaluate(self, task, evaluator_name, dataset_path):
        evaluator = self.factory.create_evaluator(task, evaluator_name)
        dataset = pd.read_csv(dataset_path)

        results = []
        for index, row in dataset.iterrows():
            if task == 'mrc':
                result = evaluator.evaluate_response(passage=row['passage'], question=row['question'], model_answer=row['answer'])
            elif task == 'summarization':
                result = evaluator.evaluate_response(context=row['user'], summary=row['assistant'])
            # Implement QA and Dialogue evaluation logic when evaluators are available
            else:
                result = None  # Placeholder for other tasks
            results.append(result)

        return results


def main():
    parser = argparse.ArgumentParser(description="Run evaluation services")
    parser.add_argument('--task', type=str, required=True, help="The task to evaluate (mrc, summarization, qa, dialogue)")
    parser.add_argument('--evaluator', type=str, required=True, help="The evaluator to use for the specified task")
    parser.add_argument('--api_key', type=str, required=True, help="The OpenAI API key")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset CSV file")

    args = parser.parse_args()

    manager = EvaluationManager(api_key=args.api_key)
    results = manager.evaluate(task=args.task, evaluator_name=args.evaluator, dataset_path=args.dataset)

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
    """
    python main.py --task 'mrc' --evaluator 'comprehensiveness' --api_key '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' --dataset 'data/lora32-llama70-summarization.csv'
    """