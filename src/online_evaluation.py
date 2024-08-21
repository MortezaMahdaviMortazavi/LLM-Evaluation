import argparse
import pandas as pd
import json
from feedback import get_feedback
from tqdm import tqdm


def save_results(responses, save_path, save_format):
    if save_format == "csv":
        pd.DataFrame(responses).to_csv(save_path, index=False)
    elif save_format == "json":
        with open(save_path, 'w') as f:
            json.dump(responses, f, indent=4)
    else:
        raise ValueError("Invalid save format specified.")

def main():
    parser = argparse.ArgumentParser(description="Perform online evaluation using specified metric.")
    parser.add_argument("--data_path", required=True, help="Path to the input CSV file containing the dataset.")
    parser.add_argument("--source_column", required=True, help="Name of the source column in the CSV file.")
    parser.add_argument("--target_column", required=True, help="Name of the target column in the CSV file.")
    parser.add_argument("--metric", choices=["comprehensiveness", "groundedness", "relevance"], required=True,
                        help="The metric to evaluate the dataset on.")
    parser.add_argument("--api_key", required=True, help="API key for accessing the OpenRouter service.")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="Model to use for the evaluation.")
    parser.add_argument("--save_path", required=True, help="Path to save the evaluation results.")
    parser.add_argument("--save_format", choices=["csv", "json"], required=True, help="Format to save the evaluation results.")

    args = parser.parse_args()

    # Load the dataset
    df = pd.read_csv(args.data_path)
    sources = df[args.source_column].tolist()
    targets = df[args.target_column].tolist()

    responses = []
    for idx in tqdm(range(len(df))):
        source = sources[idx]
        target = targets[idx]
        feedback = get_feedback(args.metric, source, target, args.api_key, args.model)
        responses.append(feedback)

    # Save the results in the specified format
    save_results(responses, args.save_path, args.save_format)

if __name__ == "__main__":
    main()
