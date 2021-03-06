import os
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union
from functools import partial
from overrides import overrides

from datasets import load_from_disk, Dataset
from transformers.utils import logging
from transformers.tokenization_utils import PreTrainedTokenizer

from .core import DataProcessor
from .prep import PREP_PIPELINE
from ...retrieval import SearchBase

logger = logging.get_logger(__name__)


def convert_examples_to_features(
    processor: DataProcessor,
    tokenizer: PreTrainedTokenizer,
    retriever: Optional[SearchBase] = None,
    topk: Optional[int] = 1,
    mode: str = "train",
):
    """Convert input data into features.
    If the retreiver is input, the examples are built by searching the top k contexts
    when not in train mode.
    Return both examples and features for post processing.
    """

    if mode == "test" and retriever is None:
        raise AttributeError

    if mode == "train":
        dataset: Dataset = processor.get_train_examples()
    elif mode == "eval":
        dataset: Dataset = processor.get_eval_examples()
    elif mode == "test":
        dataset: Dataset = processor.get_test_examples()
    else:
        raise NotImplemented

    logger.info(f"[{mode.upper()}] convert examples to features")

    prep_pipeline = PREP_PIPELINE[processor.model_args.reader_type]

    prep_fn, is_batched = prep_pipeline(tokenizer, mode, processor.data_args)

    if retriever is not None:
        eval_mode = mode == "eval"
        dataset = retriever.retrieve(dataset, topk=topk, eval_mode=eval_mode)
        prep_fn = partial(prep_fn, retriever=retriever)

    features = dataset.map(
        prep_fn,
        batched=is_batched,
        num_proc=processor.data_args.preprocessing_num_workers,
        remove_columns=dataset.column_names,
        load_from_cache_file=not processor.data_args.overwrite_cache,
    )

    return features, dataset


class OdqaProcessor(DataProcessor):
    """ Load Datasets from disk """

    def get_train_examples(self):
        dataset_path = self.data_args.dataset_path
        input_data_path = dataset_path

        input_data = load_from_disk(input_data_path)["train"]

        return input_data

    def get_eval_examples(self):
        dataset_path = self.data_args.dataset_path
        input_data_path = dataset_path
        input_data = load_from_disk(input_data_path)["validation"]
        return input_data

    def get_test_examples(self):
        dataset_path = self.data_args.dataset_path
        input_data_path = os.path.join(dataset_path, "test_dataset")
        input_data = load_from_disk(input_data_path)["validation"]
        return input_data
