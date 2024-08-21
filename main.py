import pandas as pd
from feedback import get_feedback

def remove_first_line(text):
    text = text.strip()
    lines = text.splitlines()  # Split the content into lines
    lines = [line for line in lines if line != '']
    return "\n".join(lines[1:])  # Join the lines except the first one

def main():
    """
        Metric choices:
            1 - comprehensiveness
            2 - groundedness
            3 - relevance
    """

    use_preprocess = False
    # metric = "groundedness"
    metric = "relevance"
    model = "openai/gpt-4o-mini"
    api_key = "YOUR-API-KEY"
    save_path = f"results/"

    df = pd.read_csv("data/lora32-llama70-summarization.csv")
    if use_preprocess:
        df['user'] = df['user'].apply(remove_first_line)
    
    sources = df['user'].tolist()
    targets = df['assistant'].tolist()
    source = sources[0]
    target = targets[0]
    

    ## Get feedback for the provided source and target
    feedback = get_feedback(metric, source, target, api_key, model)
    

if __name__ == "__main__":
    main()
