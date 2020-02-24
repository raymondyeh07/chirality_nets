"""Implements a Chiral GRU."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from chiral_layers.chiral_rnn_base import ChiralRnnBase


class ChiralGru(ChiralRnnBase):
  def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
               dropout=0.,
               sym_groupings=([6, 6, 1], [6, 6, 1]), neg_dim_in=1, neg_dim_out=1):
    mode = 'GRU'
    super(ChiralGru, self).__init__(mode, input_size, hidden_size,
                                        num_layers, bias, dropout,
                                        sym_groupings, neg_dim_in, neg_dim_out)

  def rnn_cell(self, x, hidden, w_ih, w_hh, b_ih, b_hh):
    gi = F.linear(x, w_ih, b_ih)
    gh = F.linear(hidden, w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = torch.sigmoid(i_r + h_r)
    inputgate = torch.sigmoid(i_i + h_i)
    newgate = torch.tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)
    return hy
