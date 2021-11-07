<h1 align="center"> Open-Domain Question Answering 👋</h1>

<!-- <p align="center">
  <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/boostcampaitech2/klue-level2-nlp-03?style=social">
  <img alt="GitHub Pipenv locked Python version" src="https://img.shields.io/github/pipenv/locked/python-version/boostcampaitech2/klue-level2-nlp-03?style=plastic">
  <img alt="Conda" src="https://img.shields.io/conda/pn/boostcampaitech2/klue-level2-nlp-03">
</p>   -->

## Overview Description

MRC is a task of evaluating model that can answer a question about a given text passage. We formulate KLUE-MRC as to predict the answer span in the given passage corresponding to the question. The input is a concatenated sequence of the question and the passage separated with a delimiter. The output is the start and end positions of the predicted answer span within the passage.


We provide three question types: paraphrase, multi-sentence reasoning, and unanswerable, in order to evaluate different aspects of machine reading capability of a model. These question types prevent a model from exploiting reasoning shortcuts with simple word-matching by enforcing lexical and syntactic variations when workers generate questions. The questions also should be answered by considering the full query sentence.



## Evaluation Methods
The evaluation metrics for KLUE-MRC are 1) exact match (EM) and 2) character-level ROUGE-W (ROUGE), which can be viewed as longest common consecutive subsequence (LCCS)-based F1 score.


EM is the most commonly used metric for QA tasks, which measures the equality of ground truth and predicted answer string. If there are multiple gold labels, a model can earn score when at least one prediction is matched.


In contrast, ROUGE gives a partial score although a model fails to predict exactly matched answer. Due to the characteristics of Korean, an answer span can be located inside of a single word, hence subword-level span should be considered. ROUGE calculates F1 score of the length ratio of LCCS to a prediction and the length ratio of LCCS to a ground truth string. In case of multiple ground-truth answer spans having the same meaning but different lexical variations (e.g. TV, Television), we use the maximum ROUGE score among the combinations of answers and the prediction. We do not adopt character-level F1 score (char F1), which is used in all the previous Korean MRC datasets, since it measures character overlap regardless of the order. When a model predicts `한국의 위인들 (great people in Korea)` and an answer is `국한된 범위 (limited scope)`, a metric should give a low score. ROUGE scores 15.38, whereas char F1 gives 54.55 due to the overlap of `한`, `국`, and `위`.



## Code Contributors

<p>
<a href="https://github.com/iamtrueline" target="_blank">
  <img x="5" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/79238023?v=4"/>
</a>
<a href="https://github.com/promisemee" target="_blank">
  <img x="74" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/31719240?v=4"/>
</a>
<a href="https://github.com/kimminji2018" target="_blank">
  <img x="143" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/74283190?v=4"/>
</a>
<a href="https://github.com/Ihyun" target="_blank">
  <img x="212" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/32431157?v=4"/>
</a>
<a href="https://github.com/sw6820" target="_blank">
  <img x="281" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/52646313?v=4"/>
</a>
<a href="https://github.com/NayoungLee-de" target="_blank">
  <img x="350" y="5" width="64" height="64" border="0" src="https://avatars.githubusercontent.com/u/69383548?v=4"/>
</a>

</p>

## Environments 

### OS
 - UBUNTU 18.04

### Requirements
```
datasets==1.5.0
transformers==4.5.0
tqdm==4.41.1
pandas==1.1.4
scikit-learn==0.24.1
konlpy==0.5.2
numpy==1.21.3
faiss-gpu==1.7.1.post2
rank_bm25==0.2.1
pororo==0.4.2
```
### Hardware
The following specs were used to create the original solution.
- GPU(CUDA) : v100 

## Reproducing Submission
To reproduct my submission without retraining, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#Dataset-Preparation)
3. [Prepare Datasets](#Prepare-Datasets)
4. [Download Baseline Codes](#Download-Baseline-Codes)
5. [Train models](#Train-models-(GPU-needed))
6. [Inference & make submission](#Inference-&-make-submission)
7. [Ensemble](#Ensemble)
8. [Wandb graphs](#Wandb-graphs)

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
$ bash ./install/install_requirements.sh
```

## Dataset Preparation
All CSV files are already in data directory.
```
# data (51.2 MB)
tar -xzf data.tar.gz
```
### Prepare Datasets
After downloading  and converting datasets and baseline codes, the data directory is structured as:
```
├── code
│   ├── assets
│   │    ├── system_assets1.png
│   │    ├── system_assets2.png
│   │    ├── train_assets.png
│   │    └── dataset.png
│   ├── install
│   │    └── install_requirements.sh
│   ├── ensemble_csv
│   │    ├── ensemble.ipynb
│   │    ├── klue-bert-base__BM5_topk_8.csv
│   │    ├── klue-bert-base__dpr_train_topk_5.csv
│   │    ├── koelectra-base__BM25_topk_5.csv
│   │    ├── roberta_cnn__batch_16__BM5_topk_5.csv
│   │    └── roberta_cnn__batch_8__BM25_topk_5.csv
│   ├── arguments.py
│   ├── bm25_retrieval.py
│   ├── inference.py
│   ├── inference_command.txt
│   ├── model.py
│   ├── readme.md
│   ├── README_en.md
│   ├── retrieval.py
│   ├── retrieval_inference.py
│   ├── run.sh
│   ├── train.py
│   ├── train_command.txt
│   ├── trainer_qa.py
│   ├── utils_qa.py
│   └── wiki_preprocess.py
└── data
    ├── test_dataset
    │    ├── dataset_dict.json
    │    └── validataion
    │          ├── dataset.arrow
    │          ├── dataset_info.json
    │          ├── indices.arrow
    │          └── state.json
    ├── train_dataset
    │          ├── train    
    │          │    ├── dataset.arrow
    │          │    ├── dataset_info.json
    │          │    ├── indices.arrow
    │          │    └── state.json
    │          ├── validation
    │          │    ├── dataset.arrow
    │          │    ├── dataset_info.json
    │          │    ├── indices.arrow
    │          │    └── state.json    
    │          └── dataset_dict.json
    └── wikipedia_documents.json

```
#### Download Baseline code
To download baseline codes, run following command. The baseline codes will be located in `/opt/ml/code`
```
$ !wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000077/data/code.tar.gz
```

#### Download Dataset
To download dataset, run following command. The dataset will be located in `/opt/ml/dataset`
```
$ !wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000077/data/data.tar.gz
``` 
### Train Models (GPU needed)
To train extractive models, run following commands.
```
$ python train.py --output_dir ./models/train_dataset --do_train
```
To train generative models, run following commands.
```
$ python generation.py --output_dir ./models/train_dataset --do_train
```

The expected training times are:

Model | GPUs | Batch Size | Training Epochs | Training Time
------------  | ------------- | ------------- | ------------- | -------------
 roberta-large + cnn | v100 | 16 | 3 | 34m 18s
 bart-base | v100 | 8 | 3 | 11m 58s
 bert-base | v100 | 16 | 5 | 25m 07s 
 koelectra-base | v100 | 16 | 3 | 15m 43s
 t-base | v100 | 8 | 3 | 9m 57s


### Inference & Make Submission
```
$ python train.py --output_dir ./outputs/train_dataset --model_name_or_path ./models/train_dataset/ --do_eval 
```

### Wandb Graphs
- Train Graphs
<p>
    <img src="https://github.com/boostcampaitech2/mrc-level2-nlp-03/blob/main/assets/train_assets.PNG">
</p>    

- System Graphs
<p>
    <img src="https://github.com/boostcampaitech2/mrc-level2-nlp-03/blob/main/assets/system_assets1.PNG">
    <img src="https://github.com/boostcampaitech2/mrc-level2-nlp-03/blob/main/assets/system_assets2.PNG">
</p>

## Reference
[KLUE-RE - Relation Extraction](https://klue-benchmark.com/tasks/72/data/description)
