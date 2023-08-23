#!/bin/bash

# activate conda env
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mott

# main
python src/track.py with exp
