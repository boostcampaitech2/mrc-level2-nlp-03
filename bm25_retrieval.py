import os
import json
import pickle
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union

from datasets import (
    Dataset,
    load_from_disk,
)

#bm25
from rank_bm25 import BM25Okapi

#mecab for noun extraction
from konlpy.tag import Mecab

#pororo for ner
from pororo import Pororo
import random

from retrieval import SparseRetrieval

class SparseRetrieval_BM25(SparseRetrieval):
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        super(SparseRetrieval_BM25, self).__init__(tokenize_fn)
        self.tokenize_fn = tokenize_fn
        self.context_path = context_path  

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts_with_title = list(
            dict.fromkeys(['#' + v['title'] + '# ' + v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로

        print(self.contexts_with_title[0])

        #Mecab
        self.mecab = Mecab()
        self.pos_list = ['NNG', 'NNP', 'SW', 'SL', 'SH', 'SN']
        self.stopwords = ['말', '것', '사람', '그', '무엇', '누구', '곳', '경우', '각자', '지', '때']

        #NER
        self.ner = Pororo(task='ner', lang='ko')
        self.keyword_list = "PERSON ARTIFACT CITY COUNTRY LOCATION PLANT ANIMAL".split()

    def get_tokenized(self, pickle_name="sparse_embedding.bin", bm25_name = "bm25.bin") -> NoReturn:

        """
        Summary:
            Tokenized dataset을 만들고 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = pickle_name
        bm25_name = bm25_name
        tokenized_path = os.path.join(self.data_path, pickle_name)
        bm25_path = os.path.join(self.data_path, bm25_name)

        if os.path.isfile(tokenized_path) and os.path.isfile(bm25_path):
            with open(tokenized_path, "rb") as file:
                self.tokenized_contexts = pickle.load(file)
            with open(bm25_path, "rb") as file:
                self.bm25 = pickle.load(file)
            print("Tokenized pickle load.")
        else:
            print("Tokenize Wikipedia")
            self.tokenized_contexts = [self.tokenize_fn(doc) for doc in self.contexts_with_title]
            self.bm25 = BM25Okapi(self.tokenized_contexts)
            with open(tokenized_path, "wb") as file:
                pickle.dump(self.tokenized_contexts, file)
            with open(bm25_path, "wb") as file:
                pickle.dump(self.bm25, file)
            print("Tokenized pickle saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1, use_mecab=False
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.tokenized_contexts is not None, "get_tokenized() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk, use_mecab=use_mecab)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts_with_title[doc_indices[i]])

            return (doc_scores, [self.contexts_with_title[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                query_or_dataset["question"], k=topk, use_mecab=use_mecab
            )
            for idx, example in enumerate(query_or_dataset):
                # context_id, context = self.postprocess_doc(example, doc_indices[idx], topk)
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts_with_title[pid] for pid in doc_indices[idx]]
                    ),
                    # "context_id": context_id,
                    # "context": context
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1, use_mecab=False) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        if use_mecab:
            query = self.preprocess_query(query)

        tokenized_query = self.tokenize_fn(query)
        result = self.bm25.get_scores(tokenized_query)        
        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result[sorted_result].tolist()[:k]
        doc_score = result[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]

        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1, use_mecab = False
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        doc_scores = []
        doc_indices = []

        for idx, query in enumerate(tqdm(queries)):
            doc_score, doc_indice = self.get_relevant_doc(query, k, use_mecab)
            doc_scores.append(doc_score)
            doc_indices.append(doc_indice)

        return doc_scores, doc_indices

    def preprocess_query(self, text):
        query = self.mecab.pos(text)
        new_query = [x[0] for x in query if x[1] in self.pos_list and x[0] not in self.stopwords]
        # query = self.ner(text)
        # new_query = [x[0] for x in query if x[1] != 'O']
        new_query = ' '.join(new_query)
        return text + ' ' + new_query

    def postprocess_doc(self, example, doc_indices, topk):
        question = example["question"]
        query = self.ner(question)
        keywords = [x[0] for x in query if x[1] not in self.keyword_list]
        if len(keywords)==0:
            context_id = doc_indices[:topk]
            context = " ".join([self.contexts_with_title[pid] for pid in doc_indices[:topk]])
        else:
            keyword = random.choice(keywords)
            new_indices = [pid for pid in doc_indices if keyword in self.contexts_with_title[pid]]
            if len(new_indices) < topk: 
                new_indices += [pid for pid in doc_indices if pid not in new_indices]

            context_id = new_indices[:topk]
            context = " ".join([self.contexts_with_title[pid] for pid in new_indices[:topk]])

        return context_id, context
