#!/bin/bash

# Path to the base directory where create_graphs.ipynb is located and folders are structured
BASE_DIR="/home/xfulop/mvi/diploma_project_gnn"

# Execute the main notebook first
# papermill "$BASE_DIR/create_graphs.ipynb" "$BASE_DIR/create_graphs.ipynb"

# Define an array of directories containing the notebooks
notebook_dirs=("gatv2conv" "sageconv" "gatedgraphconv")

# Loop through each directory and execute all notebooks within it sequentially
for dir in "${notebook_dirs[@]}"; do
    echo "Running notebooks in $dir..."
    for notebook in "$BASE_DIR/$dir"/*.ipynb; do
        # Run the notebook with papermill, overwriting the original
        papermill "$notebook" "$notebook"
    done
done

echo "All notebooks have been executed."
