"""Implements a base class for chiral RNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from chiral_layers.chiral_layer_base import ChiralLayerBase
from chiral_layers.chiral_linear import ChiralLinear


def Recurrent(inner, reverse=False):
  """Performsn recurrence. Modified from pytorch's RNN class."""
  def forward(input, hidden, w_hi, w_hh, b_ih, b_hh):
    output = []
    steps = range(input.size(0) - 1, -1, -
                  1) if reverse else range(input.size(0))
    for i in steps:
      hidden = inner(input[i], hidden, w_hi, w_hh, b_ih, b_hh)
      # hack to handle LSTM
      output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

    if reverse:
      output.reverse()
    output = torch.cat(output, 0).view(input.size(0), *output[0].size())

    return hidden, output

  return forward


class ChiralRnnBase(nn.Module):
  def __init__(self, mode, input_size, hidden_size, num_layers=1, bias=True,
               dropout=0., sym_groupings=([6, 6, 5], [6, 6, 5]),
               neg_dim_in=1, neg_dim_out=1):
    """
    """
    super(ChiralRnnBase, self).__init__()
    self.mode = mode
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bias = bias
    self.dropout = dropout

    if mode == 'LSTM':
      num_gate = 4
    if mode == 'GRU':
      num_gate = 3
    self.num_gate = num_gate

    layer_linears = nn.ModuleList([])
    for layer in range(num_layers):
      layer_input_size = input_size if layer == 0 else hidden_size
      weight_list = nn.ModuleList([])
      for k in [layer_input_size, hidden_size]:  # for w_ih and w_hh
        gate_list = nn.ModuleList([])
        for gate_num in range(num_gate):
          if gate_num == 2:  # Cell gate goes through tanh, thus odd symmetry.
            sym_ll = partial(ChiralLinear, neg_dim_out=neg_dim_out)
          else:  # Other gaes goes through sigmoid, thus need even symmetry.
            sym_ll = partial(ChiralLinear, neg_dim_out=0)
          gate_list.append(sym_ll(k, hidden_size, bias=bias,
                                  sym_groupings=sym_groupings,
                                  neg_dim_in=neg_dim_in))
        weight_list.append(gate_list)
      layer_linears.append(weight_list)
    self.layer_linears = layer_linears

  def get_rnn_big_weight_bias_layer(self, layer_num):
    weight_list = self.layer_linears[layer_num]
    w_ih, b_ih = self.get_rnn_big_weight_bias_(weight_list, 0)
    w_hh, b_hh = self.get_rnn_big_weight_bias_(weight_list, 1)
    return w_ih, w_hh, b_ih, b_hh

  def get_rnn_big_weight_bias_(self, weight_list, in_out_idx):
    gate_list = weight_list[in_out_idx]
    w_big = []
    b_big = []
    for ll in gate_list:
      ll_w, ll_b = ll.get_masked_weight()
      w_big.append(ll_w)
      b_big.append(ll_b)
    if not self.bias:
      return torch.cat(w_big, 0), None
    else:
      return torch.cat(w_big, 0), torch.cat(b_big, 0)

  def forward(self, x, hidden=None):
    next_hidden = []
    if hidden is None:
      hx, cx = torch.zeros(x.shape[1], self.hidden_size), torch.zeros(
          x.shape[1], self.hidden_size)
      if x.is_cuda:
        hx = hx.cuda()
        cx = cx.cuda()
      if self.mode == 'LSTM':
        hidden = (hx, cx)
      elif self.mode == 'GRU':
        hidden = hx
      hidden = [hidden for _ in range(self.num_layers)]
    next_hidden = []
    rnn_forward = Recurrent(self.rnn_cell)
    for i in range(self.num_layers):
      w_ih, w_hh, b_ih, b_hh = self.get_rnn_big_weight_bias_layer(i)
      hy, output = rnn_forward(x, hidden[i], w_ih, w_hh, b_ih, b_hh)
      next_hidden.append(hy)
      x = output
      if self.dropout != 0 and i < self.num_layers - 1:
        x = F.dropout(x, p=self.dropout, training=self.training, inplace=False)

    if self.mode == 'LSTM':
      next_h, next_c = zip(*next_hidden)
      next_hidden = []
      for nh, nc in zip(next_h, next_c):
        next_hidden.append((nh, nc))
    else:
      next_hidden = torch.cat(next_hidden, 0).view(
          self.num_layers, *next_hidden[0].size())
    return x, next_hidden  # next_hidden, x

  def rnn_cell(self, x, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    raise NotImplementedError("Need to override the rnn cell.")
