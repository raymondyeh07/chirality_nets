"""Implements a chiral linear layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from chiral_layers.chiral_linear_base import ChiralLinearBase


class ChiralLinear(ChiralLinearBase):
  """Implements a linear layer with chiral equivariant."""

  def __init__(self, in_channels, out_channels, bias=True,
               sym_groupings=([6, 6, 5], [6, 6, 5]),
               neg_dim_in=1, neg_dim_out=1):
    """Chiral linear layer, shape convention follows nn.Linear.
    Args:
      in_chanels (int): size of each input sample
      out_channels (int): size of each output sample
      bias (bool): If set to ``False``, the layer will not learn an additive bias.
        Default: ``True``
      sym_groupings (tuple): Tuple consisting of in/out symmetry groupings.
      neg_dim_in (int): Input dimension for `negated coordiantes'.
      neg_dim_out (int): Output dimension for `negated coordiantes'.
    """
    super(ChiralLinear, self).__init__(in_channels, out_channels,
                                       bias,
                                       sym_groupings,
                                       neg_dim_in, neg_dim_out)

    big_weight, big_bias = self.get_weight_()
    weight_mask = torch.ones_like(big_weight)

    neg_select_group = self.neg_select_group
    pos_select_group = self.pos_select_group

    if neg_dim_out > 0:
      weight_mask[neg_select_group['out'][1], self.pos_in_select] = -1
      weight_mask[neg_select_group['out'][2],
                  pos_select_group['in'][1].flatten()] = -1
      weight_mask[neg_select_group['out'][2],
                  pos_select_group['in'][2].flatten()] = 0
    weight_mask[pos_select_group['out'][1], self.neg_in_select] = -1
    weight_mask[pos_select_group['out'][2], self.neg_in_select] = 0
    self.register_buffer('weight_mask', weight_mask)

    if self.bias:
      bias_mask = torch.ones_like(big_bias)
      if neg_dim_out > 0:
        bias_mask[neg_select_group['out'][1]] = -1
        bias_mask[neg_select_group['out'][2]] = 0
      self.register_buffer('bias_mask', bias_mask)

  def get_masked_weight(self):
    big_weight, big_bias = self.get_weight_()
    big_weight = big_weight*self.weight_mask
    if big_bias is not None:
      big_bias = self.bias_mask*big_bias
    return big_weight, big_bias

  def forward(self, x):
    big_weight, big_bias = self.get_masked_weight()
    return F.linear(x, big_weight, big_bias)
