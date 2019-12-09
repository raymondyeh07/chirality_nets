"""Implements a chiral conv1d layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from chiral_layers.chiral_conv1d_base import ChiralConv1dBase


class ChiralConv1d(ChiralConv1dBase):
  """Implements a chiral conv1d layer class."""

  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
               padding=0, dilation=1, groups=1, bias=True,
               sym_groupings=([6, 6, 5], [6, 6, 5]),
               neg_dim_in=1, neg_dim_out=1):
    """Initializes a chiral conv1d layer.

      Args:
      in_channels (int): Number of channels in the input.
      out_channels (int): Number of channels produced by the convolution
      kernel_size (int or tuple): Size of the convolving kernel
      stride (int or tuple, optional): Stride of the convolution. Default: 1
      padding (int or tuple, optional): Zero-padding added to both sides of
          the input. Default: 0
      dilation (int or tuple, optional): Spacing between kernel
          elements. Default: 1
      groups (int, optional): Number of blocked connections from input
          channels to output channels. Default: 1
      bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
      sym_groupings (tuple): Tuple consisting of in/out symmetry groupings.
      neg_dim_in (int): Input dimension for `negated coordiantes'
      neg_dim_out (int): Output dimension for `negated coordiantes'
    """
    super(ChiralConv1d, self).__init__(in_channels, out_channels,
                                       kernel_size, stride, padding,
                                       dilation, groups, bias,
                                       sym_groupings, neg_dim_in, neg_dim_out)
    # Get shared weights and bias.
    big_weight, big_bias = self.get_weight_()

    # Build odd symmetry mask.
    weight_mask = torch.ones_like(big_weight)

    # Get dimensions that need negation; the `x coordiante.'
    neg_select_group = self.neg_select_group
    # Get dimensions that does not need negation; the remaing coordiantes.
    pos_select_group = self.pos_select_group

    if neg_dim_out > 0:
      weight_mask[neg_select_group['out'][1], self.pos_in_select, :] = -1
      weight_mask[neg_select_group['out'][2],
                  pos_select_group['in'][1].flatten(), :] = -1
      weight_mask[neg_select_group['out'][2],
                  pos_select_group['in'][2].flatten(), :] = 0
    weight_mask[pos_select_group['out'][1], self.neg_in_select, :] = -1
    weight_mask[pos_select_group['out'][2], self.neg_in_select, :] = 0

    # Create fixed mask for symmetry as buffers.
    self.register_buffer('weight_mask', weight_mask)
    if self.bias:
      bias_mask = torch.ones_like(big_bias)
      if neg_dim_out > 0:
        bias_mask[neg_select_group['out'][1]] = -1
        bias_mask[neg_select_group['out'][2]] = 0
      self.register_buffer('bias_mask', bias_mask)

  def forward(self, x):
    """Performs a conv1d forward pass.

    Args:
      x (Tensor): Input tensor follows pytorch's Conv1d convention.

    Returns:
     (Tensor): Output tensor follows pytorch's Conv1D convention.
    """
    big_weight, big_bias = self.get_weight_()
    out = F.conv1d(x, big_weight*self.weight_mask, None,
                   self.stride, self.padding, self.dilation, self.groups)
    if big_bias is not None:
      out = out + (self.bias_mask*big_bias).unsqueeze(0).unsqueeze(-1)
    return out
