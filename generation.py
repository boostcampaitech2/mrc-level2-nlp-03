import logging
import os
import sys

from typing import NoReturn

import nltk
from datasets import load_metric, load_from_disk, DatasetDict

from transformers import AutoConfig, AutoTokenizer, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from utils_qa import check_no_error

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

import argparse

import wandb

logger = logging.getLogger(__name__)



def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.


    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args.model_name_or_path)

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


    model_name = model_args.model_name_or_path.split('/')[
        -1] if '/' in model_args.model_name_or_path else model_args.model_name_or_path

    eval_or_train = 'eval' if training_args.do_eval else 'train'

    based = 'ext' if data_args.extractive_based else 'gen'

    wandb.init(
        project='MRC',
        name=based + '_' + (model_name) + '_' + eval_or_train + '_' + str(training_args.per_device_train_batch_size) + '_' + str(training_args.num_train_epochs),
        config=config,
        entity='bumblebe2',
        group=(model_name) + '_' + eval_or_train,
    )

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
            max_length=min(data_args.max_source_length, tokenizer.model_max_length),
            padding="max_length" if data_args.pad_to_max_length else False,
            truncation=True
        )

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
    main()