#!/bin/bash

conda env create -f environment.yml

conda init
conda activate opd

pip install flash-attn==2.8.3 --no-build-isolation
