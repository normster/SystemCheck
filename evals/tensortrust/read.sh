#!/bin/bash

MODEL=${1}

python evaluate.py --outputs_dir outputs/${MODEL} --bootstrap 10000