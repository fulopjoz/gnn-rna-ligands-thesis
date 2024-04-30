#!/bin/bash

# Navigate to the repository root
cd /home/xfulop/mvi

# Create or clear existing .gitignore
echo "" > .gitignore

# Find files larger than 50MB and add them to .gitignore
find . -size +50M | sed 's|^\./||g' >> .gitignore

echo "Large files have been added to .gitignore."
