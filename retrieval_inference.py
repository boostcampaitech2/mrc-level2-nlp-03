"""
retrieval을 inference하기 위한 코드
"""

import json
import random
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, NoReturn, Any, Optional, Union

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

from datasets import (
    load_from_disk,
)

from transformers import AutoTokenizer

# from retrieval import SparseRetrieval
from bm25_retrieval import SparseRetrieval_BM25

# 난수 고정
def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)

def main():
    data_args = DataTrainingArguments
    model_args = ModelArguments
    training_args = TrainingArguments
    data_path = "../data"
    context_path = "wikipedia_documents_cleaned.json",
    dataset = "../data/train_dataset"
    org_dataset = load_from_disk(dataset)
    
    tokenizer = AutoTokenizer.from_pretrained(
        "klue/roberta-large",
        use_fast=False,
    )

    full_ds = org_dataset["validation"]

    retriever = SparseRetrieval_BM25(
        tokenize_fn=tokenizer.tokenize, data_path=data_path, context_path=context_path
    )
    retriever.get_tokenized()
    
    topK_list = [1, 5, 10, 20]
    result_dict = {}

    for topK in tqdm(topK_list):
        result_retriever = retriever.retrieve(full_ds, topk=topK, use_mecab=True)
        correct = 0
        for index in range(len(result_retriever)):
            # original_context = re.sub('\\n', '', result_retriever['original_context'][index])
            # original_context = re.sub('\\\\n', '', original_context)
            if  result_retriever['original_context'][index] in result_retriever['context'][index]:
                correct += 1
        result_dict[topK] = correct/len(result_retriever)

    print(result_dict)

if __name__=="__main__":
    set_seed(42)
    main()