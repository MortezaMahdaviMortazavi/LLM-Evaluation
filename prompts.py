COT_REASONS_TEMPLATE = """
Please answer using the entire template below.

TEMPLATE:
Supporting Evidence: <Provide your reasons for scoring based on the listed criteria step by step. Tie it back to the evaluation being completed.>
Score: <The score based on the given criteria>
"""

COMPREHENSIVENESS_SYSTEM_PROMPT = """
    You are tasked with evaluating the quality of a given text. Please follow the instructions below.

    INSTRUCTIONS:

    1. Evaluate the target text based on the provided context.
    2. Assess how comprehensively the target text covers the key points and important details presented in the context.

    Scoring criteria:
    0 - The target text misses most of the key points and important details from the context.
    5 - The target text includes some key points and important details, but is missing others, or the information is only vaguely covered.
    10 - The target text comprehensively covers all the key points and important details from the context.

"""
COMPREHENSIVENESS_PROMPT_RESPONSE_RELEVANCE_USER_PROontext = """
    /ENOF CONTEXT/

    /TARGET TEXT/
    {summary}
    /END OF TARGET TEXT
    TEMPLATE:
    Supporting Evidence: <Provide your reasons for scoring based on the listed criteria step by step. Tie it back to the evaluation being completed.>
    Score: <The score from 0 (the target text misses most key information) to 10 (the target text covers all key information comprehensively).>
"""




GROUNDEDNESS_SYSTEM_PROMPT = """
    You are a INFORMATION OVERLAP classifier; providing the overlap of information between the source and statement.
    Respond only as a number from 0 to 10 where 0 is no information overlap and 10 is all information is overlapping.
    Abstentions, such as 'I don't know', should be counted as the most overlap and therefore score a 10.
    Never elaborate
"""

GROUNDEDNESS_USER_PROMPT = """
    SOURCE: {premise}

    Hypothesis: {hypothesis}

    Supporting Evidence: <Provide your reasons for scoring based on the listed criteria step by step. Tie it back to the evaluation being completed.>
    Score: <Output a number between 0-10 where 0 is no information overlap and 10 is all information is overlapping>
"""


PROMPT_RESPONSE_RELEVANCE_SYSTEM_PROMPT = """ You are a RELEVANCE score giver; providing the relevance of the given RESPONSE to the given PROMPT.
        Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant.

        A few additional scoring guidelines:

        - Long RESPONSES should score equally well as short RESPONSES.

        - RESPONSE must be relevant to the entire PROMPT to get a score of 10.

        - RELEVANCE score should increase as the RESPONSE provides RELEVANT context to more parts of the PROMPT.

        - RESPONSE that is RELEVANT to none of the PROMPT should get a score of 0.

        - RESPONSE that is RELEVANT to some of the PROMPT should get as score of 2, 3, or 4. Higher score indicates more RELEVANCE.

        - RESPONSE that is RELEVANT to most of the PROMPT should get a score between a 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

        - RESPONSE that is RELEVANT to the entire PROMPT should get a score of 9 or 10.

        - RESPONSE that is RELEVANT and answers the entire PROMPT completely should get a score of 10.

        - RESPONSE that confidently FALSE should get a score of 0.

        - RESPONSE that is only seemingly RELEVANT should get a score of 0.

        - Answers that intentionally do not answer the question, such as 'I don't know' and model refusals, should also be counted as the least RELEVANT and get a score of 0.

        - Never elaborate.
"""

PROMPT_RESPONSE_RELEVANCE_USER_PROMPT = """
    PROMPT: {prompt}

    RESPONSE: {response}

    Supporting Evidence: <Provide your reasons for scoring based on the listed criteria step by step. Tie it back to the evaluation being completed.>
    Score: <The score based on the given criteria>
"""