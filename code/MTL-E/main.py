import logging
import torch
import nltk
import numpy as np
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from tqdm import tqdm as tqdm1

import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import set_seed
from transformers.file_utils import is_offline_mode
from utils.arguments import parse_args
from multitask_model import MultitaskModel
from preprocess import convert_to_features
from multitask_data_collator import MultitaskTrainer, NLPDataCollator
from multitask_eval import multitask_eval_fn
from checkpoint_model import save_model
from pathlib import Path


logger = logging.getLogger(__name__)


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def main():
    args = parse_args()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    dataset_dict = {
        "hyperbole": load_dataset(
            "multitask_dataloader.py",
            data_files={
#                 "train": "../../data/hyperbole_mtl/train_101_8.csv",
#                 "validation": "../../data/hyperbole_mtl/val_101_8.csv",
                "train": str(args.htrain_file),
                "validation": str(args.hvalidation_file),
            },
        ),
        "metaphor": load_dataset(
            "multitask_dataloader.py",
            data_files={
#                 "train": "../../data/metaphor_mtl/train_101_8.csv",
#                 "validation": "../../data/metaphor_mtl/val_101_8.csv",
                "train": str(args.mtrain_file),
                "validation": str(args.mvalidation_file),
            },
        ),
#         "irony": load_dataset(
#             "multitask_dataloader.py",
#             data_files={
#                 "train": "../../data/mtl_data/irony/train.csv",
#                 "validation": "../../data/mtl_data/irony/test.csv",
#             },
#         ),
#         "sarcasm": load_dataset(
#             "multitask_dataloader.py",
#             data_files={
#                 "train": "../../data/mtl_data/sarcasm/train.csv",
#                 "validation": "../../data/mtl_data/sarcasm/test.csv",
#             },
#         ),
    }

    for task_name, dataset in dataset_dict.items():
        print(task_name)
        print(dataset_dict[task_name]["train"][0])
        print()

    model_names = [args.model_name_or_path] * 2
    config_files = model_names
#     for idx, task_name in enumerate(["hyperbole", "metaphor","irony","sarcasm"]):
    for idx, task_name in enumerate(["hyperbole", "metaphor"]):
        model_file = Path(f"./{task_name}_model/pytorch_model.bin")
        config_file = Path(f"./{task_name}_model/config.json")
        if model_file.is_file():
            model_names[idx] = f"./{task_name}_model"

        if config_file.is_file():
            config_files[idx] = f"./{task_name}_model"
    
    print(model_names)

    multitask_model = MultitaskModel.create(
        model_name=model_names[0],
        model_type_dict={
            "hyperbole": transformers.AutoModelForSequenceClassification,
            "metaphor": transformers.AutoModelForSequenceClassification,
#             "irony": transformers.AutoModelForSequenceClassification,
#             "sarcasm": transformers.AutoModelForSequenceClassification,
        },
        model_config_dict={
            "hyperbole": transformers.AutoConfig.from_pretrained(
                model_names[0], num_labels=2
            ),
            "metaphor": transformers.AutoConfig.from_pretrained(
                model_names[1], num_labels=2
            ),
#             "irony": transformers.AutoConfig.from_pretrained(
#                 model_names[2], num_labels=2
#             ),
#             "sarcasm": transformers.AutoConfig.from_pretrained(
#                 model_names[3], num_labels=2
#             ),
        },
    )

#     print(multitask_model.encoder.embeddings.word_embeddings.weight.data_ptr())
#     print(
#         multitask_model.taskmodels_dict[
#             "hyperbole"
#         ].roberta.embeddings.word_embeddings.weight.data_ptr()
#     )
#     print(
#         multitask_model.taskmodels_dict[
#             "metaphor"
#         ].roberta.embeddings.word_embeddings.weight.data_ptr()
#     )
#     print(
#         multitask_model.taskmodels_dict[
#             "irony"
#         ].roberta.embeddings.word_embeddings.weight.data_ptr()
#     )
#     print(
#         multitask_model.taskmodels_dict[
#             "sarcasm"
#         ].roberta.embeddings.word_embeddings.weight.data_ptr()
#     )

    convert_func_dict = {
        "hyperbole": convert_to_features,
        "metaphor": convert_to_features,
#         "irony": convert_to_features,
#         "sarcasm": convert_to_features,
    }

    columns_dict = {
        "hyperbole": ["input_ids", "attention_mask", "labels"],
        "metaphor": ["input_ids", "attention_mask", "labels"],
#         "irony": ["input_ids", "attention_mask", "labels"],
#         "sarcasm": ["input_ids", "attention_mask", "labels"],
    }

    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
                batched=True,
                load_from_cache_file=False,
            )
            print(
                task_name,
                phase,
                len(phase_dataset),
                len(features_dict[task_name][phase]),
            )
            features_dict[task_name][phase].set_format(
                type="torch",
                columns=columns_dict[task_name],
            )
            print(
                task_name,
                phase,
                len(phase_dataset),
                len(features_dict[task_name][phase]),
            )

    train_dataset = {
        task_name: dataset["train"] for task_name, dataset in features_dict.items()
    }

    trainer = MultitaskTrainer(
        model=multitask_model,
        args=transformers.TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            learning_rate=1e-5,
            do_train=True,
            num_train_epochs=args.num_train_epochs,
            # Adjust batch size if this doesn't fit on the Colab GPU
            per_device_train_batch_size=args.per_device_train_batch_size,
            save_steps=3000,
        ),
        data_collator=NLPDataCollator(),
        train_dataset=train_dataset,
    )
    trainer.train()

    ## evaluate on given tasks
    multitask_eval_fn(multitask_model, args.model_name_or_path, dataset_dict)

    ## save model for later use
#     save_model(args.model_name_or_path, multitask_model)


if __name__ == "__main__":
    main()
