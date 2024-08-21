#!/bin/bash

# Bash script to run online and/or offline evaluation

# Gather the shared input CSV path
echo "Please enter the path to your input CSV file:"
read data_path

# Flags to determine if evaluations should be run
run_online=false
run_offline=false

# Collect information for online evaluation if desired
echo "Do you want to perform online evaluation? (yes/no)"
read online_evaluation

if [ "$online_evaluation" == "yes" ]; then
    run_online=true
    echo "Please enter the source column name:"
    read source_column
    echo "Please enter the target column name:"
    read target_column
    echo "Please enter the metric (comprehensiveness, groundedness, relevance):"
    read metric
    echo "Please enter your API key:"
    read api_key

    echo "In which format would you like to save the online evaluation results? (csv/json)"
    read save_format

    if [[ "$save_format" != "csv" && "$save_format" != "json" ]]; then
        echo "Invalid format specified. Please choose either 'csv' or 'json'."
        exit 1
    fi

    echo "Please specify the path where the online evaluation results should be saved (excluding the file extension):"
    read online_save_path_base

    online_save_path="${online_save_path_base}.${save_format}"
fi

# Collect information for offline evaluation if desired
echo "Do you want to perform offline evaluation? (yes/no)"
read offline_evaluation

if [ "$offline_evaluation" == "yes" ]; then
    run_offline=true
    echo "Please specify the path where the offline evaluation log should be saved (including the file extension, e.g., .log):"
    read offline_logfile_path
    echo "Please enter the name of the label column (reference):"
    read label_column
    echo "Please enter the name of the prediction column:"
    read prediction_column
    echo "Please enter the metrics to evaluate (e.g., ExactMatch F1Score BLEUScore):"
    read -a metrics
    echo "Please enter the model name for the tokenizer and metrics (or press enter to use default):"
    read model_name

    if [ -z "$model_name" ]; then
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct"
    fi
fi

# Execute online evaluation if requested
if [ "$run_online" == true ]; then
    python online_evaluator.py --data_path $data_path --source_column $source_column --target_column $target_column --metric $metric --api_key $api_key --save_path $online_save_path --save_format $save_format
fi

# Execute offline evaluation if requested
if [ "$run_offline" == true ]; then
    python offline_evaluator.py --input_csv $data_path --label_column $label_column --prediction_column $prediction_column --logfile $offline_logfile_path --model_name $model_name --metrics "${metrics[@]}"
fi

# If neither online nor offline evaluation was selected
if [ "$run_online" != true ] && [ "$run_offline" != true ]; then
    echo "No evaluation was selected."
fi
