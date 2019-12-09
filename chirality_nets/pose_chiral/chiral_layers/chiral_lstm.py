"""Implements a Chiral LSTM."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from chiral_layers.chiral_rnn_base import ChiralRnnBase


class ChiralLstm(ChiralRnnBase):
  def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
               dropout=0.,
               sym_groupings=([6, 6, 1], [6, 6, 1]),
               neg_dim_in=1, neg_dim_out=1):
    mode = 'LSTM'
    super(ChiralLstm, self).__init__(mode, input_size, hidden_size,
                                     num_layers, bias, dropout,
                                     sym_groupings, neg_dim_in, neg_dim_out)

  def rnn_cell(self, x, hidden, w_ih, w_hh, b_ih, b_hh):
    hx, cx = hidden
    # Only need one bias.
    gates = F.linear(x, w_ih, None) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    hy = cellgate
    cy = cellgate
    return hy, cy
