#!/bin/bash

source /home/amira/miniconda3/bin/activate base

# activate base

jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0