from typing import Optional,Tuple, Union

import torch
import torch.nn as nn

from transformers.modeling_utils import PoolerAnswerClass, PoolerStartLogits,  PoolerEndLogits, SquadHeadOutput

from .modeling_utils import (
    QAConvSDSLayer,
    ConvLayer,
    AttentionLayer
)


class QAConvSDSHead(nn.Module):
    """QA conv SDS head"""

    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        n_layers: int,
        num_labels: int,
    ):
        """
        Args:
            input_size (int): max sequence lengths
            hidden_dim (int): backbone's hidden dimension
            n_layers (int): number of layers
            num_labels (int): number of labels used for QA
        """
        super().__init__()
        convs = []
        for n in range(n_layers):
            convs.append(QAConvSDSLayer(input_size, hidden_dim))
        self.convs = nn.Sequential(*convs)
        self.qa_output = nn.Linear(hidden_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape == (bsz, seq_length, hidden_dim)
        out = self.convs(x)
        return self.qa_output(out) # (bsz, seq_length, hidden_dim)


class QAConvHeadWithAttention(nn.Module):
    """QA conv head with attention"""

    def __init__(self, config):
        """
        Args:
            config (ModelArguments): ModelArguments
        """
        super().__init__()
        self.attention = AttentionLayer(config)
        self.conv = ConvLayer(config)
        self.classify_layer = nn.Linear(
            config.qa_conv_out_channel*3, 2, bias=True)

    def forward(self, x, token_type_ids):
        """
        Args:
            x (torch.Tensor): Head input
            token_type_ids (torch.Tensor): Token type ids of input_ids

        Returns:
            torch.Tensor: output logits (batch_size * max_seq_legth * 2)
        """
        embedded_value = self.attention(x, token_type_ids)
        concat_output = self.conv(embedded_value)
        logits = self.classify_layer(concat_output)
        return logits


class QAConvHead(nn.Module):
    """Simple QA conv head"""

    def __init__(self, config):
        """
        Args:
            config (ModelArguments): ModelArguments
        """
        super().__init__()
        self.conv = ConvLayer(config)
        self.classify_layer = nn.Linear(
            config.qa_conv_out_channel*3, 2, bias=True)

    def forward(self, **kwargs):
        """
        Args:
            **kwargs: x, input_ids, sep_token_id
        Returns:
            torch.Tensor: output logits (batch_size * max_seq_legth * 2)
        """
        concat_output = self.conv(kwargs['x'])
        logits = self.classify_layer(concat_output)
        return logits


class SQuADHead(nn.Module):
    r"""
    A SQuAD head inspired by XLNet.
    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model, will be used to grab the :obj:`hidden_size` of the model and the
            :obj:`layer_norm_eps` to use.
    """

    def __init__(self, config):
        super().__init__()
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        cls_index: Optional[torch.LongTensor] = None,
        is_impossible: Optional[torch.LongTensor] = None,
        p_mask: Optional[torch.FloatTensor] = None,
        return_dict: bool = False,
    ) -> Union[SquadHeadOutput, Tuple[torch.FloatTensor]]:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                Final hidden states of the model on the sequence tokens.
            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Positions of the first token for the labeled span.
            end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Positions of the last token for the labeled span.
            cls_index (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Position of the CLS token for each sentence in the batch. If :obj:`None`, takes the last token.
            is_impossible (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Whether the question has a possible answer in the paragraph or not.
            p_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len)`, `optional`):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.
            return_dict (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        Returns:
        """
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
                total_loss += cls_loss * 0.5

            return SquadHeadOutput(loss=total_loss) if return_dict else (total_loss,)

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = nn.functional.softmax(start_logits, dim=-1)  # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = nn.functional.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_states = torch.einsum("blh,bl->bh", hidden_states, start_log_probs)
            cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)

            if not return_dict:
                return (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits)
            else:
                return SquadHeadOutput(
                    start_top_log_probs=start_top_log_probs,
                    start_top_index=start_top_index,
                    end_top_log_probs=end_top_log_probs,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )