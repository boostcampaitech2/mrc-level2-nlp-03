#!/bin/bash
### install requirements for pstage3 baseline
# pip requirements
pip install datasets==1.5.0
pip install transformers==4.5.0
pip install tqdm==4.41.1
pip install pandas==1.1.4
pip install scikit-learn==0.24.1
pip install konlpy==0.5.2
pip install numpy==1.21.3

# faiss install (if you want to)
pip install faiss-gpu

# install wandb
pip install wandb

# install knockknock
pip install knockknock

#install rank_bm25
pip install rank_bm25

#install Pororo
pip install pororo
