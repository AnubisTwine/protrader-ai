#!/bin/bash
# Ensure we are in the project root
cd "$(dirname "$0")"

# Activate the virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found in .venv/"
    exit 1
fi

# Add current directory to PYTHONPATH to ensure local modules are found
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run streamlit using the module syntax to ensure it uses the venv interpreter
python -m streamlit run dashboard/app.py --server.enableCORS false --server.enableXsrfProtection false
