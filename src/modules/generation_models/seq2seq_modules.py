"""
@Desc:
@Reference:
@Notes:
"""
"""
@Desc:
@Reference:
- from plan-and-write
https://bitbucket.org/VioletPeng/language-model/src/master/
- seq2seq attention
https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/master/
3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
@Notes:
"""
import logging
import math
from typing import Optional
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers import BartTokenizer
from transformers.models.bart.modeling_bart import (
    BartModel, shift_tokens_right, Seq2SeqModelOutput, BartConfig, BartPretrainedModel, BartDecoder,
    CrossEntropyLoss, Seq2SeqLMOutput, BartAttention, BartEncoder, BartDecoderLayer, _expand_mask,
    BartForConditionalGeneration, BaseModelOutput
)

logger = logging.getLogger(__name__)


class Seq2SeqEncoder(nn.Module):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None,
                 enc_hid_dim=512, dec_hid_dim=512, n_layers=3, dropout=0.2):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.n_layers = n_layers

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        self.dropout = dropout
        self.rnn = nn.GRU(embed_dim, enc_hid_dim, n_layers,
                           dropout=self.dropout,
                           batch_first=True,
                           bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)  # 2 for bidirection

    def forward(self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1] # [batch, seq]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # mask pad tokens
        input_ids *= attention_mask

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        inputs_embeds = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # outputs are always from the top hidden layer
        outputs, hidden_states = self.rnn(inputs_embeds)

        dec_hidden = torch.tanh(self.fc(torch.cat((hidden_states[-2, :, :], hidden_states[-1, :, :]), dim=1)))
        dec_hidden = nn.functional.dropout(dec_hidden, p=self.dropout, training=self.training)  # [batch, dec_hid_dim]

        # hidden = [batch size, dec hid dim]
        if not return_dict:
            return tuple(v for v in [outputs, dec_hidden] if v is not None)
        return BaseModelOutput(
            last_hidden_state=outputs, hidden_states=dec_hidden, attentions=None
        )


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.val = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, enc_hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        src_len = encoder_outputs.shape[1]
        # repeat decoder hidden state src_len times
        enc_hidden = enc_hidden.unsqueeze(1).repeat(1, src_len, 1) # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        energy = torch.tanh(self.attn(torch.cat((enc_hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, dec hid dim]
        attention = self.val(energy).squeeze(2)
        # attention= [batch size, src len]
        return torch.softmax(attention, dim=1)


class Seq2SeqDecoder(nn.Module):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None, attention=None,
                 enc_hid_dim=512, dec_hid_dim=512, dropout=0.2, teacher_forcing_ratio=0.5, device=None):
        super().__init__()
        if attention is None:
            attention = Attention(enc_hid_dim=enc_hid_dim, dec_hid_dim=dec_hid_dim)
        self.padding_idx = config.pad_token_id
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embed_dim = config.d_model
        self.output_dim = embed_dim
        self.attention = attention
        self.dropout = dropout
        self.rnn = nn.GRU((enc_hid_dim * 2) + embed_dim, dec_hid_dim,
                           dropout=self.dropout,
                           batch_first=True,
                           bidirectional=False)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + embed_dim, embed_dim)
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def decode(self, input, dec_hidden, encoder_outputs):
        # input = [batch size]
        # enc_hidden = [batch size, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        # retrieve input_ids and inputs_embeds

        input = input.unsqueeze(1) # input = [batch size, 1]

        embedded = nn.functional.dropout(self.embed_tokens(input) * self.embed_scale, p=self.dropout,
                                         training=self.training) # embedded = [batch, 1, size, emb dim]

        attn = self.attention(dec_hidden, encoder_outputs)

        attn = attn.unsqueeze(1) # a = [batch size, 1, src len]

        weighted = torch.bmm(attn, encoder_outputs) # weighted = [batch size, 1, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2) # rnn_input = [batch size, 1, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hx=dec_hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [batch size, 1, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden.permute(1, 0, 2)).all()

        embedded = embedded.squeeze(1)  # [batch, embed]
        output = output.squeeze(1)  # [batch, dec_hid_dim]
        weighted = weighted.squeeze(1)  # [batch, (enc hid dim * 2)]

        # [(enc_hid_dim * 2) + dec_hid_dim + dec_hid_dim, embed_dim]
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]
        # hidden = [batch size, dec hid dim]
        return prediction, hidden.squeeze(0)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                enc_hidden=None,
                enc_outputs=None,
                ):
        if attention_mask is not None:
            input_ids *= attention_mask

        batch_size = input_ids.shape[0]
        target_len = input_ids.shape[1]
        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len, self.output_dim).to(self.device)  # [batch, seq, embed]

        # first input to the decoder is the <sos> tokens, bart end_token: 2
        input = input_ids[:, 0] # [batch_size, 1]

        for idx in range(0, target_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, enc_hidden = self.decode(input, enc_hidden, enc_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[:, idx] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < self.teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            if idx+1 < target_len:
                input = input_ids[:, idx+1] if teacher_force else top1
        # output[idx] should be input_ids[idx+1]
        return outputs

# to suit bart decoder
class Seq2SeqEncoderForBart(nn.Module):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None,
                 enc_hid_dim=1024, n_layers=3, dropout=0.2):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.n_layers = n_layers

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        self.dropout = dropout
        self.rnn = nn.GRU(embed_dim, enc_hid_dim, n_layers,
                           dropout=self.dropout,
                           batch_first=True,
                           bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2 + embed_dim, embed_dim)  # 2 for bidirection

    def forward(self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1] # [batch, seq]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # mask pad tokens
        input_ids *= attention_mask

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        inputs_embeds = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)
        # outputs = [batch size, src len, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # outputs are always from the top hidden layer
        outputs, hidden_states = self.rnn(inputs_embeds)
        outputs = nn.functional.dropout(outputs, p=self.dropout, training=self.training)
        outputs = torch.cat([outputs, inputs_embeds], dim=-1)
        outputs = self.fc(outputs) # [batch, seq, embed]
        outputs = nn.functional.dropout(outputs, p=self.dropout, training=self.training)
        # hidden = [batch size, dec hid dim]
        if not return_dict:
            return tuple(v for v in [outputs, hidden_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=outputs, hidden_states=hidden_states, attentions=None
        )