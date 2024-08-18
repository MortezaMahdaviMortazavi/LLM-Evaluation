from abc import ABC, abstractmethod
from textwrap import dedent
from typing import ClassVar

class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators. Defines the common interface and behavior.
    """

    system_prompt: ClassVar[str]
    user_prompt: ClassVar[str]

    @abstractmethod
    def generate_prompt(self, passage: str, question: str, model_answer: str) -> str:
        """
        Generate the complete prompt to be fed into the model.

        :param passage: The source passage text.
        :param question: The question based on the passage.
        :param model_answer: The answer provided by the model.
        :return: The complete prompt string.
        """
        pass


class AccuracyCompletenessEvaluator(BaseEvaluator):
    """
    Evaluator for assessing the accuracy and completeness of model-generated responses.
    """

    system_prompt: ClassVar[str] = dedent(
        """
        You are tasked with strictly evaluating the accuracy and completeness of the answer provided for a reading comprehension task. Please follow the instructions below.

        INSTRUCTIONS:

        1. Compare the model's answer with the correct answer derived from the passage.
        2. Score the answer based on its accuracy and completeness.

        Scoring criteria:
        0 - The answer is completely incorrect, irrelevant, or contradictory to the passage.
        1-2 - The answer contains minor elements that are correct but is mostly incorrect or irrelevant.
        3 - The answer has some correct components but is significantly incomplete or inaccurate.
        4 - The answer is incomplete or inaccurate but shows a basic understanding of the passage.
        5 - The answer is somewhat accurate but misses important details or includes minor inaccuracies.
        6-7 - The answer is mostly accurate but lacks detail or contains slight errors.
        8 - The answer is accurate and covers most details, but minor information may be missing.
        9 - The answer is very accurate with only trivial omissions or slight lack of detail.
        10 - The answer is fully accurate and comprehensive, including all relevant details from the passage.

        TEMPLATE:
        Score: <The score from 0 (completely incorrect) to 10 (fully accurate and comprehensive).>
        Accuracy and Completeness Assessment: <Provide a detailed assessment of how accurate and complete the answer is compared to the correct information from the passage.>
        """
    )

    user_prompt: ClassVar[str] = dedent(
        """
        /PASSAGE/
        {passage}
        /END OF PASSAGE/

        /QUESTION/
        {question}
        /END OF QUESTION/

        /MODEL ANSWER/
        {model_answer}
        /END OF MODEL ANSWER/
        """
    )

    @staticmethod
    def generate_prompt(passage: str, question: str, model_answer: str) -> str:
        return AccuracyCompletenessEvaluator.system_prompt + "\n\n" + AccuracyCompletenessEvaluator.user_prompt.format(
            passage=passage,
            question=question,
            model_answer=model_answer
        )



class RelevanceJustificationEvaluator(BaseEvaluator):
    """
    Evaluator for assessing the relevance and justification of model-generated responses.
    """

    system_prompt: ClassVar[str] = dedent(
        """
        You are tasked with evaluating the relevance and justification of the answer provided for a reading comprehension task. Please follow the instructions below.

        INSTRUCTIONS:

        1. Assess how relevant the model's answer is to the question asked.
        2. Evaluate whether the answer is justified based on the passage content.

        Scoring criteria:
        0 - The answer is completely irrelevant or does not address the question at all.
        1-2 - The answer is mostly irrelevant but may contain a slight connection to the question.
        3 - The answer is partially relevant but lacks sufficient connection or justification.
        4 - The answer is somewhat relevant with minimal justification or slight off-topic elements.
        5 - The answer is relevant but lacks full justification or contains minor irrelevant details.
        6-7 - The answer is mostly relevant and justified, with minor off-topic elements or missing justification.
        8 - The answer is relevant and well-justified, with only minor areas lacking.
        9 - The answer is highly relevant and justified, with very slight deviations or minor omissions.
        10 - The answer is fully relevant and thoroughly justified by the passage content.

        TEMPLATE:
        Score: <The score from 0 (irrelevant) to 10 (highly relevant and justified).>
        Relevance and Justification Assessment: <Provide a detailed assessment of how relevant and justified the answer is in relation to the question and the passage content.>
        """
    )

    user_prompt: ClassVar[str] = dedent(
        """
        /PASSAGE/
        {passage}
        /END OF PASSAGE/

        /QUESTION/
        {question}
        /END OF QUESTION/

        /MODEL ANSWER/
        {model_answer}
        /END OF MODEL ANSWER/
        """
    )

    @staticmethod
    def generate_prompt(passage: str, question: str, model_answer: str) -> str:
        return RelevanceJustificationEvaluator.system_prompt + "\n\n" + RelevanceJustificationEvaluator.user_prompt.format(
            passage=passage,
            question=question,
            model_answer=model_answer
        )




class DepthUnderstandingEvaluator(BaseEvaluator):
    """
    Evaluator for assessing the depth of understanding demonstrated by model-generated responses.
    """

    system_prompt: ClassVar[str] = dedent(
        """
        You are tasked with evaluating the depth of understanding demonstrated by the model's answer in a reading comprehension task. Please follow the instructions below.

        INSTRUCTIONS:

        1. Evaluate how deeply the model's answer reflects an understanding of the passage's content.
        2. Consider whether the answer captures underlying implications or nuances.

        Scoring criteria:
        0 - The answer shows no understanding of the passage.
        1-2 - The answer shows minimal understanding with significant gaps or misunderstandings.
        3 - The answer reflects a superficial understanding with limited depth.
        4 - The answer shows basic understanding but misses important nuances or implications.
        5 - The answer demonstrates basic understanding but lacks depth or misses crucial details.
        6-7 - The answer reflects a good understanding with some nuances captured, but with minor gaps.
        8 - The answer demonstrates a deep understanding with most nuances captured, though some minor elements may be missing.
        9 - The answer shows a very deep understanding with only slight omissions of nuances or details.
        10 - The answer demonstrates a profound and thorough understanding, capturing all key nuances and implications.

        TEMPLATE:
        Score: <The score from 0 (no understanding) to 10 (deep and thorough understanding).>
        Depth of Understanding Assessment: <Provide a detailed assessment of the depth of understanding demonstrated by the answer, including any nuances or implications captured.>
        """
    )

    user_prompt: ClassVar[str] = dedent(
        """
        /PASSAGE/
        {passage}
        /END OF PASSAGE/

        /QUESTION/
        {question}
        /END OF QUESTION/

        /MODEL ANSWER/
        {model_answer}
        /END OF MODEL ANSWER/
        """
    )

    @staticmethod
    def generate_prompt(passage: str, question: str, model_answer: str) -> str:
        return DepthUnderstandingEvaluator.system_prompt + "\n\n" + DepthUnderstandingEvaluator.user_prompt.format(
            passage=passage,
            question=question,
            model_answer=model_answer
        )



class PrecisionConcisenessEvaluator(BaseEvaluator):
    """
    Evaluator for assessing the precision and conciseness of model-generated responses.
    """

    system_prompt: ClassVar[str] = dedent(
        """
        You are tasked with evaluating the precision and conciseness of the model's answer in a reading comprehension task. Please follow the instructions below.

        INSTRUCTIONS:

        1. Evaluate how precise and concise the model's answer is.
        2. Consider whether the answer includes unnecessary details or omits critical information.

        Scoring criteria:
        0 - The answer is imprecise or verbose, with significant irrelevant details or omissions.
        1-2 - The answer is largely imprecise or overly verbose, with some minor correct elements.
        3 - The answer is somewhat precise but includes irrelevant details or omits important information.
        4 - The answer is fairly precise but could be clearer or more concise with minor irrelevant details.
        5 - The answer is moderately precise and concise, with some unnecessary details or minor omissions.
        6-7 - The answer is mostly precise and concise, with minor irrelevant details or slight lack of clarity.
        8 - The answer is precise and concise, with only minor improvements possible.
        9 - The answer is very precise and concise, with very slight areas for improvement.
        10 - The answer is highly precise and concise, delivering all necessary information without any irrelevant details.

        TEMPLATE:
        Score: <The score from 0 (not precise or concise) to 10 (highly precise and concise).>
        Precision and Conciseness Assessment: <Provide a detailed assessment of how precise and concise the answer is, highlighting any unnecessary details or omissions.>
        """
    )

    user_prompt: ClassVar[str] = dedent(
        """
        /PASSAGE/
        {passage}
        /END OF PASSAGE/

        /QUESTION/
        {question}
        /END OF QUESTION/

        /MODEL ANSWER/
        {model_answer}
        /END OF MODEL ANSWER/
        """
    )

    @staticmethod
    def generate_prompt(passage: str, question: str, model_answer: str) -> str:
        return PrecisionConcisenessEvaluator.system_prompt + "\n\n" + PrecisionConcisenessEvaluator.user_prompt.format(
            passage=passage,
            question=question,
            model_answer=model_answer
        )





class MRC_ComprehensiveEvaluator(BaseEvaluator):
    """
    Evaluator for comprehensive evaluation of model-generated responses in reading comprehension tasks.
    """

    system_prompt: ClassVar[str] = dedent(
        """
        You are tasked with evaluating a model-generated response for a reading comprehension task. Your evaluation should be holistic, considering the response across multiple dimensions to determine how well it meets the following criteria:

        INSTRUCTIONS:

        1. Accuracy and Completeness:
           - Assess whether the response accurately reflects the information provided in the passage. Consider if it covers all necessary details and correctly represents the content without introducing errors or omissions.

        2. Relevance and Justification:
           - Evaluate the relevance of the response to the specific question asked. Determine if the response directly addresses the question with pertinent information and whether it is well-supported by the content of the passage.

        3. Depth of Understanding:
           - Judge the depth of understanding shown in the response. Consider if the response captures the underlying nuances, implications, and key themes of the passage, indicating a thorough grasp of the material.

        4. Precision and Conciseness:
           - Analyze the precision and conciseness of the response. Assess whether the response is clear, focused, and free of unnecessary details or verbosity, while still conveying all essential information.

        SCORING GUIDELINES:

        - 0-2: The response is fundamentally flawed, showing major inaccuracies, irrelevance, or a lack of understanding. It may also be overly verbose or vague, failing to address the question adequately.
        - 3-4: The response shows some correct elements but is incomplete, partially irrelevant, or lacks sufficient depth. It may contain errors or be imprecise, with unnecessary or missing details.
        - 5-6: The response is generally accurate and relevant but may miss some important details or nuances. It demonstrates a basic understanding and is mostly clear and concise, though improvements are needed.
        - 7-8: The response is accurate, relevant, and demonstrates a good understanding of the passage. It covers most key details and nuances, is well-justified, and is both precise and concise, with only minor issues.
        - 9-10: The response is exemplary, fully accurate, and highly relevant. It demonstrates a deep and nuanced understanding, covers all important details, is well-justified, and is exceptionally clear, precise, and concise.

        TEMPLATE:
        Score: <Provide a single score from 0 to 10, considering all evaluation criteria.>
        Overall Assessment: <Explain how the response meets or falls short of the criteria. Discuss its accuracy, relevance, depth of understanding, and precision/concision.>
        """
    )

    user_prompt: ClassVar[str] = dedent(
        """
        /PASSAGE/
        {passage}
        /END OF PASSAGE/

        /QUESTION/
        {question}
        /END OF QUESTION/

        /MODEL ANSWER/
        {model_answer}
        /END OF MODEL ANSWER/
        """
    )

    @staticmethod
    def generate_prompt(passage: str, question: str, model_answer: str) -> str:
        """
        Generate the complete prompt to be fed into the model.

        :param passage: The source passage text.
        :param question: The question based on the passage.
        :param model_answer: The answer provided by the model.
        :return: The complete prompt string.
        """
        return MRC_ComprehensiveEvaluator.system_prompt + "\n\n" + MRC_ComprehensiveEvaluator.user_prompt.format(
            passage=passage,
            question=question,
            model_answer=model_answer
        )
    

