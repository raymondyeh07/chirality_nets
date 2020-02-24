"""Implements a chiral linear base class."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from chiral_layers.chiral_layer_base import ChiralLayerBase


class ChiralLinearBase(ChiralLayerBase):
  """Implements a base class for chiral linear."""

  def __init__(self, in_channels, out_channels, bias=True,
               sym_groupings=([6, 6, 5], [6, 6, 5]),
               neg_dim_in=1, neg_dim_out=1):
    """Chiral linear base layer, shape convention follows nn.Linear.
    Args:
      in_chanels (int): size of each input sample
      out_channels (int): size of each output sample
      bias (bool): If set to ``False``, the layer will not learn an additive bias.
        Default: ``True``
      sym_groupings (tuple): Tuple consisting of in/out symmetry groupings.
      neg_dim_in (int): Input dimension for `negated coordiantes'.
      neg_dim_out (int): Output dimension for `negated coordiantes'.
    """
    super(ChiralLinearBase, self).__init__(in_channels, out_channels,
                                           sym_groupings,
                                           neg_dim_in, neg_dim_out)
    self.in_channels = in_channels
    self.out_channels = out_channels

    joint_in_dim = self.joint_in_dim
    joint_out_dim = self.joint_out_dim

    # Define layer parameters.
    self.w1 = nn.Parameter(torch.Tensor(sym_groupings[1][0]*joint_out_dim,
                                        sym_groupings[0][0]*joint_in_dim))
    self.w2 = nn.Parameter(torch.Tensor(sym_groupings[1][1]*joint_out_dim,
                                        sym_groupings[0][1]*joint_in_dim))
    self.w3 = nn.Parameter(torch.Tensor(sym_groupings[1][0]*joint_out_dim,
                                        sym_groupings[0][2]*joint_in_dim))
    self.w4 = nn.Parameter(torch.Tensor(sym_groupings[1][2]*joint_out_dim,
                                        sym_groupings[0][0]*joint_in_dim))
    self.w5 = nn.Parameter(torch.Tensor(sym_groupings[1][2]*joint_out_dim,
                                        sym_groupings[0][2]*joint_in_dim))
    self.bias = bias

    if bias:
      self.b1 = nn.Parameter(torch.Tensor(sym_groupings[1][0]*joint_out_dim))
      self.b2 = nn.Parameter(torch.Tensor(sym_groupings[1][-1]*joint_out_dim))
    else:
      self.register_parameter('b1', None)
      self.register_parameter('b2', None)
    self.reset_parameters()

  def reset_parameters(self,):
    """Initializes parameters."""
    # Matches LSMT initialization.
    bound = 1.0 / math.sqrt(self.out_channels)
    with torch.no_grad():
      nn.init.uniform_(self.w1, -bound, bound)
      nn.init.uniform_(self.w2, -bound, bound)
      nn.init.uniform_(self.w3, -bound, bound)
      nn.init.uniform_(self.w4, -bound, bound)
      nn.init.uniform_(self.w5, -bound, bound)

      if self.bias:
        nn.init.uniform_(self.b1, -bound, bound)
        nn.init.uniform_(self.b2, -bound, bound)

  def get_weight_(self,):
    """Forms the parameter matrix and bias with sharing."""
    r1 = torch.cat([self.w1, self.w2, self.w3], 1)
    r2 = torch.cat([self.w2, self.w1, self.w3], 1)
    r3 = torch.cat([self.w4, self.w4, self.w5], 1)
    ret = torch.cat([r1, r2, r3], 0)
    if self.bias:
      ret_bias = torch.cat([self.b1, self.b1, self.b2], 0)
    else:
      ret_bias = None
    return ret, ret_bias

  def forward(self, x):
    """Performs a forward pass of a linear layer."""
    big_weight, big_bias = self.get_weight_()
    return F.linear(x, big_weight, big_bias)
