from unicodedata import bidirectional
import transformers
from params import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import AutoModel
from memory_profiler import profile, memory_usage
from typing import Optional

class TModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(
            pretrained_model_name_or_path="roberta-base", cache_dir=MODEL_CACHE_DIR, config=config)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(True)
        self.plain_ner = nn.Linear(config.hidden_size, len(LABEL_BIO))
        if not BASELINE:
            self.boundary_encoder = nn.LSTM(bidirectional=True, input_size=config.hidden_size, hidden_size=LSTM_HIDDEN, batch_first=True)
            self.boundary_decoder = nn.LSTM(bidirectional=False, input_size=LSTM_HIDDEN*2, hidden_size=LSTM_HIDDEN, batch_first=True)
            self.boundary_biaffine = BoundaryBiaffine(LSTM_HIDDEN, LSTM_HIDDEN*2, 1)
            self.boundary_seg = BoundarySeg()
            self.boundary_final0 = nn.Linear(config.hidden_size, LSTM_HIDDEN*2)
            self.boundary_final1 = nn.Linear(LSTM_HIDDEN*2, LSTM_HIDDEN*2)
            self.boundary_fc = nn.Linear(LSTM_HIDDEN*2, len(BOUNDARY_LABEL))
            self.seg_final0 = nn.Linear(config.hidden_size, LSTM_HIDDEN*4)
            self.seg_final1 = nn.Linear(LSTM_HIDDEN*4, LSTM_HIDDEN*4)
            self.boundary = nn.ModuleList([self.boundary_encoder, self.boundary_decoder, self.boundary_biaffine, self.boundary_final0, self.boundary_final1, self.boundary_fc])
            self.type_lstm = nn.LSTM(bidirectional=True, input_size=config.hidden_size, hidden_size=LSTM_HIDDEN, batch_first=True)
            self.type_final0 = nn.Linear(config.hidden_size, LSTM_HIDDEN*2)
            self.type_final1 = nn.Linear(LSTM_HIDDEN*2, LSTM_HIDDEN*2)
            self.type_fc = nn.Linear(LSTM_HIDDEN*2, len(LABEL_2_ID))
            self.type_predict = nn.ModuleList([self.type_lstm, self.type_final0, self.type_final1, self.type_fc])
            self.ner_final = nn.Linear(LSTM_HIDDEN*8+config.hidden_size, len(LABEL_BIO))
            self.ner = nn.ModuleList([self.seg_final0, self.seg_final1, self.ner_final])
        self.get_trigram = nn.Conv1d(LSTM_HIDDEN*2, LSTM_HIDDEN*2, 3, padding=1, bias=False)
        self.get_trigram.weight = torch.nn.Parameter(torch.ones([LSTM_HIDDEN*2, LSTM_HIDDEN*2, 3]), requires_grad=False)
        self.get_trigram.requires_grad_ = False
        
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        outputs = self.transformer(
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
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        if BASELINE:
            ner_result = self.plain_ner(sequence_output)
            return ner_result
        else:
            boundary_hidden = self.boundary_encoder(sequence_output)[0]
            seg_result = self.get_trigram(boundary_hidden.transpose(1,2)).transpose(1,2)
            seg_result = self.boundary_decoder(seg_result)[0]
            seg_result = F.softmax(self.boundary_biaffine(seg_result, boundary_hidden), dim=2)
            seg_result = self.boundary_seg(seg_result, boundary_hidden)
            boundary_result = F.logsigmoid(self.boundary_final0(sequence_output)+self.boundary_final1(boundary_hidden)).mul(boundary_hidden)
            type_hidden = self.type_lstm(sequence_output)[0]
            type_result = F.logsigmoid(self.type_final0(sequence_output)+self.type_final1(type_hidden)).mul(type_hidden)
            ner_result = F.logsigmoid(self.seg_final0(sequence_output)+self.seg_final1(seg_result)).mul(seg_result)
            ner_result = self.ner_final(torch.cat([sequence_output, boundary_result, type_result, seg_result], dim=-1))
            del seg_result, boundary_result, type_result
            torch.cuda.empty_cache()
            return ner_result, self.boundary_fc(boundary_hidden), self.type_fc(type_hidden)


class BoundarySeg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, span_adjacency, bound_hidden):
        temp = []
        for j in range(MAX_LEN):
            j_sum = []
            for i in range(j, MAX_LEN):
                result = torch.cat([bound_hidden[:, i], bound_hidden[:, j]], 1)
                result = result * span_adjacency[:, j, i]
                j_sum.append(result)
            temp.append(torch.sum(torch.stack(j_sum, dim=0), dim=0))
        return torch.stack(temp, 1)


class PairwiseBilinear(nn.Module):
    """ A bilinear module that deals with broadcasting for efficient memory usage.
    Input: tensors of sizes (N x L1 x D1) and (N x L2 x D2)
    Output: tensor of size (N x L1 x L2 x O)"""
    def __init__(self, input1_size, input2_size, output_size, bias=True):
        super().__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.output_size = output_size

        self.weight = nn.Parameter(torch.zeros(input1_size, input2_size, output_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_size), requires_grad=True) if bias else 0

    def forward(self, input1, input2):
        input1_size = list(input1.size())
        input2_size = list(input2.size())
        output_size = [input1_size[0], input1_size[1], input2_size[1], self.output_size]

        # ((N x L1) x D1) * (D1 x (D2 x O)) -> (N x L1) x (D2 x O)
        intermediate = torch.mm(input1.contiguous().view(-1, input1_size[-1]), self.weight.view(-1, self.input2_size * self.output_size))
        # (N x L2 x D2) -> (N x D2 x L2)
        input2 = input2.transpose(1, 2)
        # (N x (L1 x O) x D2) * (N x D2 x L2) -> (N x (L1 x O) x L2)
        output = intermediate.view(input1_size[0], input1_size[1] * self.output_size, input2_size[2]).bmm(input2)
        # (N x (L1 x O) x L2) -> (N x L1 x L2 x O)
        output = output.view(input1_size[0], input1_size[1], self.output_size, input2_size[1]).transpose(2, 3).contiguous()
        # (N x L1 x L2 x O) + (O) -> (N x L1 x L2 x O)
        output = output + self.bias

        return output

class BoundaryBiaffine(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        self.W_bilin = PairwiseBilinear(input1_size, input2_size, output_size)
        self.U = nn.Linear(input1_size, output_size)
        self.V = nn.Linear(input2_size, output_size)

    def forward(self, input1, input2):
        return self.W_bilin(input1, input2).add(self.U(input1).unsqueeze(2)).add(self.V(input2).unsqueeze(1))

class FocalLoss(torch.nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.log_softmax = torch.nn.LogSoftmax(-1)
        self.nll_loss = torch.nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return 0.
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = self.log_softmax(x)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss