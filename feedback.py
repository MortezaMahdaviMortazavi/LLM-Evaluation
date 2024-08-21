import argparse
import json
import requests

from prompts import (
    COT_REASONS_TEMPLATE, GROUNDEDNESS_SYSTEM_PROMPT, 
    GROUNDEDNESS_USER_PROMPT, 
    COMPREHENSIVENESS_SYSTEM_PROMPT, 
    COMPREHENSIVENESS_USER_PROMPT, 
    PROMPT_RESPONSE_RELEVANCE_SYSTEM_PROMPT, 
    PROMPT_RESPONSE_RELEVANCE_USER_PROMPT
)


def request_api(prompt, api_key, model='openai/gpt-4o-mini', temperature=1e-5) -> dict:
    """
    Sends a prompt to the GPT model via the OpenRouter API and retrieves the generated response.
    """
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            data=json.dumps({
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )

        response.raise_for_status()
        response_json = response.json()
        generated_text = response_json['choices'][0]['message']['content']
        response_dict = parse_response(generated_text)
        return response_dict

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None



def parse_response(response: str) -> dict:
    """
    Parse the API response into a structured format.
    """
    import re
    score_match = re.search(r"Score:\s*(\d+)", response)
    score = int(score_match.group(1)) if score_match else None
    assessment_start = response.find("Supporting Evidence:")
    if assessment_start != -1:
        assessment = response[assessment_start + len("Supporting Evidence:"):].strip()
    else:
        assessment = response.split("Score:")[1].strip() if "Score:" in response else response

    return {
        "score": score,
        "supporting_evidence": assessment
    }



def generate_prompt(metric, source, target):
    """
    Generates a specific prompt for the chosen metric based on provided source and target.
    """
    if metric == "comprehensiveness":
        prompt_template = f"""
        {COT_REASONS_TEMPLATE}
        {COMPREHENSIVENESS_SYSTEM_PROMPT}
        {COMPREHENSIVENESS_USER_PROMPT.format(context=source, target=target)}
        """
    elif metric == "groundedness":
        prompt_template = f"""
        {COT_REASONS_TEMPLATE}
        {GROUNDEDNESS_SYSTEM_PROMPT}
        {GROUNDEDNESS_USER_PROMPT.format(premise=source, hypothesis=target)}
        """
    elif metric == "relevance":
        prompt_template = f"""
        {COT_REASONS_TEMPLATE}
        {PROMPT_RESPONSE_RELEVANCE_SYSTEM_PROMPT}
        {PROMPT_RESPONSE_RELEVANCE_USER_PROMPT.format(prompt=source, response=target)}
        """
    else:
        raise ValueError("Invalid metric provided. Choose from 'comprehensiveness', 'groundedness', or 'relevance'.")

    return prompt_template.strip()


def get_feedback(metric:str, source:str, target:str, api_key:str, model:str = 'openai/gpt-4o-mini'):
    """
    Generates feedback for a single source and target pair based on the given metric.
    """
    prompt_content = generate_prompt(metric, source=source, target=target)
    result = request_api(prompt_content, api_key, model=model)
    return result