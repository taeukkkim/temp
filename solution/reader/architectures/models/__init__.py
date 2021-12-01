from .modeling_bert import *
from .modeling_electra import *
from .modeling_roberta import *


from transformers import (
    AutoModelForQuestionAnswering as AutoQA,
)


class AutoModelForQuestionAnswering(AutoQA):
    """ Base class for Extractive Model. """
    reader_type: str = "extractive"

    def __init__(self, config):
        super().__init__(config)
        assert config.reader_type == self.reader_type
