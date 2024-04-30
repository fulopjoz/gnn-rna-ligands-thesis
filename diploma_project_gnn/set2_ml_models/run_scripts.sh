#!/bin/bash

# Define the base directory
base_dir="/home/xfulop/mvi/diploma_project_gnn/set2_ml_models"

# Loop through each model folder
for model_dir in "$base_dir"/*; do
    # Check if it's a directory
    if [ -d "$model_dir" ]; then
        # Loop through each algorithm folder
        for algo_dir in "$model_dir"/*; do
            # Check if it's a directory
            if [ -d "$algo_dir" ]; then
                # Change directory to the algorithm folder
                cd "$algo_dir" || exit

                # Run the shell script if it exists
                if [ -f "run_notebooks_$(basename "$algo_dir").sh" ]; then
                    echo "Running script in $algo_dir"
                    bash "run_notebooks_$(basename "$algo_dir").sh"
                else
                    echo "No script found in $algo_dir"
                fi
            fi
        done
    fi
done
