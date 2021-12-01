from typing import Optional,Tuple

import torch

from transformers import (
    ElectraForQuestionAnswering as QA,
    ElectraModel,
    ElectraPreTrainedModel
)

from transformers.file_utils import ModelOutput
from transformers.models.electra.configuration_electra import ElectraConfig
from transformers.modeling_outputs import (
    QuestionAnsweringModelOutput,
)

from ..modeling_heads import QAConvSDSHead, SQuADHead


class ElectraForQuestionAnswering(QA):
    reader_type: str = "extractive"

    def __init__(self, config):
        super().__init__(config)
        assert config.reader_type == self.reader_type


class ElectraForQAWithConvSDSHead(QA):
    reader_type: str = "extractive"

    def __init__(self, config):
        # Initialize on ElectraPreTrainedModel
        super(ElectraForQuestionAnswering, self).__init__(config)
        assert config.reader_type == self.reader_type

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        self.qa_outputs = QAConvSDSHead(
            config.qa_conv_input_size,
            config.hidden_size,
            config.qa_conv_n_layers,
            config.num_labels,
        )

        self.init_weights()


class ElectraForQuestionAnsweringOutput(ModelOutput):
    """
    Base class for outputs of question answering models using a :obj:`SquadHead`.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned if both :obj:`start_positions` and :obj:`end_positions` are provided):
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification
            losses.
        start_top_log_probs (``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        start_top_index (``torch.LongTensor`` of shape ``(batch_size, config.start_n_top)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Indices for the top config.start_n_top start token possibilities (beam-search).
        end_top_log_probs (``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Log probabilities for the top ``config.start_n_top * config.end_n_top`` end token possibilities
            (beam-search).
        end_top_index (``torch.LongTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Indices for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        cls_logits (``torch.FloatTensor`` of shape ``(batch_size,)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Log probabilities for the ``is_impossible`` label of the answers.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    start_top_log_probs: Optional[torch.FloatTensor] = None
    start_top_index: Optional[torch.LongTensor] = None
    end_top_log_probs: Optional[torch.FloatTensor] = None
    end_top_index: Optional[torch.LongTensor] = None
    cls_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ElectraForQuestionAnsweringSquad(ElectraPreTrainedModel):
    config_class = ElectraConfig
    base_model_prefix = "electra"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        self.qa_outputs = SQuADHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        is_impossible=None,
        cls_index=None,
        p_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = discriminator_hidden_states[0]

        outputs = self.qa_outputs(
            sequence_output,
            start_positions=start_positions,
            end_positions=end_positions,
            cls_index=cls_index,
            is_impossible=is_impossible,
            p_mask=p_mask,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs + discriminator_hidden_states[1:]

        return ElectraForQuestionAnsweringOutput(
            loss=outputs.loss,
            start_top_log_probs=outputs.start_top_log_probs,
            start_top_index=outputs.start_top_index,
            end_top_log_probs=outputs.end_top_log_probs,
            end_top_index=outputs.end_top_index,
            cls_logits=outputs.cls_logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )