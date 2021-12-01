from transformers import Trainer

from .mixin import ToMixin


class BaseTrainer(Trainer, ToMixin):
    pass
