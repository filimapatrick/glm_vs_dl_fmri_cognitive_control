#!/usr/bin/env bash

# Set FSL environment if needed
if [ -z "$FSLDIR" ]; then
    export FSLDIR="/Users/patrickfilima/fsl"
    export PATH="$FSLDIR/bin:$PATH"
    . ${FSLDIR}/etc/fslconf/fsl.sh
fi

# Ensure virtual environment is activated if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "fmri_env" ]; then
    source fmri_env/bin/activate
fi

# Run the python script with all passed arguments
python3 "$(dirname "$0")/02_preprocessing.py" "$@"
