from transformers import PreTrainedModel

from .core import ReaderBase
from .architectures import (
    AutoModelForQuestionAnswering,
)
from .trainers import (
    BaseTrainer,
    QuestionAnsweringTrainer,
)


class ExtractiveReader(ReaderBase):
    reader_type: str = "extractive"
    default_model: PreTrainedModel = AutoModelForQuestionAnswering
    default_trainer: BaseTrainer = QuestionAnsweringTrainer
