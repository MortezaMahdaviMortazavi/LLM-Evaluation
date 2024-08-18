import openai
from abc import ABC, abstractmethod
from typing import Type, Any
from prompts import MRCBaseEvaluator, AccuracyCompletenessEvaluator, RelevanceJustificationEvaluator, DepthUnderstandingEvaluator, PrecisionConcisenessEvaluator

class EvaluationFeedbackBase(ABC):
    """
    Base class for task evaluation services. It interacts with the OpenAI API
    and can be extended for specific tasks such as MRC, summarization, translation, etc.
    """

    def __init__(self, api_key: str, evaluator: Type[MRCBaseEvaluator]):
        """
        Initialize the evaluation service with the API key and evaluator class.

        :param api_key: OpenAI API key for making requests.
        :param evaluator: An instance of the evaluator class to be used.
        """
        self.api_key = api_key
        self.evaluator = evaluator
        openai.api_key = self.api_key

    def evaluate_response(self, **kwargs) -> dict:
        """
        Evaluate the response using the selected evaluator.

        :param kwargs: The specific inputs required for the evaluation task (e.g., passage, question, model_answer).
        :return: A dictionary containing the evaluation score and assessment.
        """
        prompt = self.generate_prompt(**kwargs)
        response = self._call_openai_api(prompt)
        return self._parse_response(response)

    @abstractmethod
    def generate_prompt(self, **kwargs) -> str:
        """
        Generate the prompt based on the specific task and inputs provided.
        Must be implemented by derived classes.

        :param kwargs: The specific inputs required for the task.
        :return: The complete prompt string to be sent to the model.
        """
        pass

    def _call_openai_api(self, prompt: str) -> str:
        """
        Call the OpenAI API with the generated prompt.

        :param prompt: The complete prompt string to send to the API.
        :return: The raw response from the API.
        """
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].text.strip()

    def _parse_response(self, response: str) -> dict:
        """
        Parse the API response into a structured format.

        :param response: The raw response from the OpenAI API.
        :return: A dictionary containing the score and assessment.
        """
        lines = response.split("\n")
        score_line = next((line for line in lines if line.startswith("Score:")), None)
        assessment_line = next((line for line in lines if line.startswith("Assessment:")), None)

        score = int(score_line.split(":")[1].strip()) if score_line else None
        assessment = assessment_line.split(":")[1].strip() if assessment_line else response

        return {
            "score": score,
            "assessment": assessment
        }

class MRCEvaluationService(EvaluationFeedbackBase):
    """
    Evaluation service for MRC (Machine Reading Comprehension) tasks.
    """

    def generate_prompt(self, passage: str, question: str, model_answer: str) -> str:
        """
        Generate the prompt specifically for MRC tasks.
        """
        return self.evaluator.generate_prompt(passage, question, model_answer)

class SummarizationEvaluationService(EvaluationFeedbackBase):
    """
    Evaluation service for Summarization tasks.
    """

    def generate_prompt(self, context: str, summary: str) -> str:
        """
        Generate the prompt specifically for Summarization tasks.
        """
        return self.evaluator.generate_prompt(context=context, summary=summary)


class QA_EvaluationService(EvaluationFeedbackBase):
    """
    Evaluation service for Question Answering tasks.
    """

    def generate_prompt(self, context: str, question: str, answer: str) -> str:
        """
        Generate the prompt specifically for Question Answering tasks.
        """
        raise NotImplementedError("Question Answering evaluation service is not implemented yet.")

class DialogueEvaluationService(EvaluationFeedbackBase):
    """
    Evaluation service for Dialogue tasks.
    """

    def generate_prompt(self, dialogue_history: str, response: str) -> str:
        """
        Generate the prompt specifically for Dialogue tasks.
        """
        raise NotImplementedError("Dialogue evaluation service is not implemented yet.")

# Example usage:
if __name__ == "__main__":
    api_key = "your_openai_api_key"
    
    # MRC Evaluation Example
    accuracy_evaluator = AccuracyCompletenessEvaluator()
    mrc_service = MRCEvaluationService(api_key=api_key, evaluator=accuracy_evaluator)
    result = mrc_service.evaluate_response(
        passage="The Amazon rainforest, often referred to as the lungs of the planet...",
        question="What role does the Amazon rainforest play in the Earth’s atmosphere?",
        model_answer="The Amazon rainforest produces 20% of the world’s oxygen."
    )
    print("MRC Evaluation Result:", result)
    
    # Summarization Evaluation Example (context and summary)
    summarization_service = SummarizationEvaluationService(api_key=api_key, evaluator=accuracy_evaluator)
    try:
        result = summarization_service.evaluate_response(
            context="Summarize the key points about the Amazon rainforest...",
            summary="The Amazon rainforest produces 20% of the world’s oxygen."
        )
        print("Summarization Evaluation Result:", result)
    except NotImplementedError as e:
        print(e)
