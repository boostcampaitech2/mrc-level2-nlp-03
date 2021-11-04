import logging
import os
import sys

from typing import List, Callable, NoReturn, NewType, Any
import dataclasses

import nltk
from datasets import load_metric, load_from_disk, Dataset, DatasetDict

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from tokenizers import Tokenizer
from tokenizers.models import WordPiece

from utils_qa import postprocess_qa_predictions, check_no_error
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

import argparse

import wandb

# import fire

# from knockknock import slack_sender
# webhook_url = "https://hooks.slack.com/services/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"
# @slack_sender(webhook_url=webhook_url, channel="#nlp-wandb")

logger = logging.getLogger(__name__)



def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.


    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args.model_name_or_path)

    # model_args.model_name_or_path = args.model_name
    # model_name_or_path = model_args.model_name_or_path.split('/')[-1] if '/' in model_args.model_name_or_path else model_args.model_name_or_path

    # [참고] argument를 manual하게 수정하고 싶은 경우에 아래와 같은 방식을 사용할 수 있습니다
    # training_args.per_device_train_batch_size = 4
    # print(training_args.per_device_train_batch_size)

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name is not None else model_args.model_name_or_path,

        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        # use_fast=True,
        config=config,
    )
    print(f'tokenizer : {model_args.model_args.config_name if model_args.config_name is not None else model_args.model_name_or_path}')
    print(tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
    print(model)
    # training_args = training_args(
    #     output_dir = args.output_dir,
    #     do_train = args.do_train,
    #     do_eval = args.do_eval,
    #     evaluation_strategy = args.evaluation_strategy,
    #     per_device_train_batch_size = args.per_device_train_batch_size,
    #     per_device_eval_batch_size = args.per_device_eval_batch_size,
    #     # learing_rate = args.learing_rate,
    #     weight_decay = args.weight_decay,
    #     num_train_epochs = args.num_train_epochs,
    #     # learning_rate = args.learning_rate,
    #     lr_scheduler_type = args.lr_scheduler_type,
    #     # dataloader_num_workers = args.dataloader_num_workers,
    #     # metric_for_best_model = args.metric_for_best_model,
    #     # greater_is_better = args.greater_is_better,
    #     label_smoothing_factor = args.label_smoothing_factor,
    #     logging_dir=args.logging_dir,
    #     logging_steps=args.logging_steps,
    #     eval_steps=args.eval_steps,
    #     load_best_model_at_end=args.load_best_model_at_end,
    # )

    model_name = model_args.model_name_or_path.split('/')[
        -1] if '/' in model_args.model_name_or_path else model_args.model_name_or_path

    eval_or_train = 'eval' if training_args.do_eval else 'train'

    based='gen'
    wandb.init(
        project='MRC',
        name=based + '_' + (model_name) + '_' + eval_or_train + '_' + str(training_args.per_device_train_batch_size) + '_' + str(training_args.num_train_epochs),
        config=config,
        entity='bumblebe2',
        group=(model_name) + '_' + eval_or_train,
    )

    # wandb.config.update()parser.parse_args_into_dataclasses())

    print(
        type(training_args),
        type(model_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )

    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        run_gen_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_gen_mrc(
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        model_args: ModelArguments,
        datasets: DatasetDict,
        tokenizer,
        model,
) -> NoReturn:
    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Train preprocessing / 전처리를 진행합니다.
    def preprocess_function(examples):
        inputs = [f"question: {q}  context: {c} </s>" for q, c in zip(examples["question"], examples["context"])]
        targets = [f'{a["text"][0]} </s>' for a in examples['answers']]
        model_inputs = tokenizer(
            inputs,
            max_length=min(data_args.max_source_length, tokenizer.model_max_length),
            padding="max_length" if data_args.pad_to_max_length else False,
            truncation=True
        )

        # targets(label)을 위해 tokenizer 설정
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=data_args.max_target_length,
                padding="max_length" if data_args.pad_to_max_length else False,
                truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["example_id"] = []
        for i in range(len(model_inputs["labels"])):
            model_inputs["example_id"].append(examples["id"][i])
        return model_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]

        # dataset에서 train feature를 생성합니다.
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
        )

    # Validation preprocessing
    def prepare_validation_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        inputs = [f"question: {q}  context: {c} </s>" for q, c in zip(examples["question"], examples["context"])]
        targets = [f'{a["text"][0]} </s>' for a in examples['answers']]
        tokenized_examples = tokenizer(
            # examples[question_column_name if pad_on_right else context_column_name],
            # examples[context_column_name if pad_on_right else question_column_name],
            # truncation="only_second" if pad_on_right else "only_first",
            # max_length=data_args.max_seq_length,
            # stride=data_args.doc_stride,
            # return_overflowing_tokens=True,
            # return_offsets_mapping=True,
            # # return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            # padding="max_length" if data_args.pad_to_max_length else False,
            max_length=min(data_args.max_source_length, tokenizer.model_max_length),
            padding="max_length" if data_args.pad_to_max_length else False,
            truncation=True
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []

        # for i in range(len(tokenized_examples["input_ids"])):
        #     # sequence id를 설정합니다 (to know what is the context and what is the question).
        #     sequence_ids = tokenized_examples.sequence_ids(i)
        #     context_index = 1 if pad_on_right else 0
        #
        #     # 하나의 example이 여러개의 span을 가질 수 있습니다.
        #     sample_index = sample_mapping[i]
        #     tokenized_examples["example_id"].append(examples["id"][sample_index])
        #
        #     # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
        #     tokenized_examples["offset_mapping"][i] = [
        #         (o if sequence_ids[k] == context_index else None)
        #         for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        #     ]
        return tokenized_examples

    if training_args.do_eval:
        # 전체 데이터로 평가
        eval_examples = datasets["validation"]

        # Validation Feature 생성
        eval_dataset = eval_examples.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=False,
        )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
    )

    # Post-processing:

    def postprocess_text(preds, labels):
        """
        postprocess는 nltk를 이용합니다.
        Huggingface의 TemplateProcessing을 사용하여
        정규표현식 기반으로 postprocess를 진행할 수 있지만
        해당 미션에서는 nltk를 이용하여 간단한 후처리를 진행합니다
        """

        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # decoded_labels은 rouge metric을 위한 것이며, f1/em을 구할 때 사용되지 않음
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 간단한 post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        formatted_predictions = [{"id": ex["id"], "prediction_text": decoded_preds[i]} for i, ex in
                                 enumerate(datasets["validation"].select(range(data_args.max_val_samples)))]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in
                      datasets["validation"].select(range(data_args.max_val_samples))]

        result = metric.compute(predictions=formatted_predictions, references=references)
        return result

    args = Seq2SeqTrainingArguments(
        output_dir=training_args.output_dir,
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
        predict_with_generate=True,#training_args.,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        num_train_epochs=training_args.num_train_epochs,
        save_strategy='epoch',
        save_total_limit=2  # 모델 checkpoint를 최대 몇개 저장할지 설정
    )

    metric = load_metric("squad")
    # Trainer 초기화
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # train_result = trainer.train(resume_from_checkpoint=None)



    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
    #
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
    #
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    #
        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
    #
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=data_args.max_target_length,
            num_beams=data_args.num_beams,
            metric_key_prefix="eval"
        )

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # args
    # parser.add_argument('--output_dir', type=str, default="./outputs/train_dataset/ ")
    # parser.add_argument('--do_train', type=bool, default=False)
    # parser.add_argument('--do_eval', type=bool, default=False)
    # parser.add_argument('--save_total_limit', type=int, default=5)
    # parser.add_argument('--model_name_or_path', type=str, default='klue/bert-base')
    # parser.add_argument('--save_steps', type=int, default=500)
    # parser.add_argument('--num_train_epochs', type=int, default=5)

    # parser.add_argument('--learning_rate', type=float, default=5e-5)

    # parser.add_argument('--lr_scheduler_type', type=str, default='constant_with_warmup')

    # parser.add_argument('--dataloader_num_workers', type=int, default=8)
    # parser.add_argument('--metric_for_best_model', type=int, default=8)
    # parser.add_argument('--greater_is_better', type=int, default=8)

    # parser.add_argument('--label_smoothing_factor', type=float, default=0.1)
    # parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    # parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    # parser.add_argument('--warmup_steps', type=int, default=500)
    # parser.add_argument('--weight_decay', type=float, default=0.01)
    # parser.add_argument('--logging_dir', type=str, default="./logs")
    # parser.add_argument('--logging_steps', type=int, default=100)
    # parser.add_argument('--evaluation_strategy', type=str, default="steps")
    # parser.add_argument('--eval_steps', type=int, default=500)
    # parser.add_argument('--load_best_model_at_end', type=bool, default=True)

    # metric_for_best_model = args.metric_for_best_model,
    # greater_is_better = args.greater_is_better,
    # label_smoothing_factor = args.label_smoothing_factor,
    # logging_dir = args.logging_dir,
    # logging_steps = args.logging_steps,
    # eval_steps = args.eval_steps,
    # load_best_model_at_end = args.load_best_model_at_end,

    # args = parser.parse_args()

    # fire.Fire(main)
    main()


    # print(args)
