"""Implements a chiral conv1d base class."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from chiral_layers.chiral_layer_base import ChiralLayerBase


class ChiralConv1dBase(ChiralLayerBase):
  """Implements a base class for chiral conv1d."""

  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
               padding=0, dilation=1, groups=1, bias=True,
               sym_groupings=([6, 6, 5], [6, 6, 5]),
               neg_dim_in=1, neg_dim_out=1):
    """Initializes a chiral conv1d base class.

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
      bias (bool, optional): If ``True``, adds a learnable bias to the output.
          Default: ``True``
      sym_groupings (tuple): Tuple consisting of in/out symmetry groupings.
      neg_dim_in (int): Input dimension for `negated coordiantes'.
      neg_dim_out (int): Output dimension for `negated coordiantes'.
    """
    super(ChiralConv1dBase, self).__init__(
        in_channels, out_channels, sym_groupings, neg_dim_in, neg_dim_out)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.bias = bias

    assert groups == 1  # Currently do not support groupings.

    joint_out_dim = self.joint_out_dim
    joint_in_dim = self.joint_in_dim

    # Define layer parameters and bias.
    self.w1 = nn.Parameter(torch.Tensor(sym_groupings[1][0]*joint_out_dim,
                                        sym_groupings[0][0]*joint_in_dim,
                                        kernel_size))
    self.w2 = nn.Parameter(torch.Tensor(sym_groupings[1][1]*joint_out_dim,
                                        sym_groupings[0][1]*joint_in_dim,
                                        kernel_size))
    self.w3 = nn.Parameter(torch.Tensor(sym_groupings[1][0]*joint_out_dim,
                                        sym_groupings[0][2]*joint_in_dim,
                                        kernel_size))
    self.w4 = nn.Parameter(torch.Tensor(sym_groupings[1][2]*joint_out_dim,
                                        sym_groupings[0][0]*joint_in_dim,
                                        kernel_size))
    self.w5 = nn.Parameter(torch.Tensor(sym_groupings[1][2]*joint_out_dim,
                                        sym_groupings[0][2]*joint_in_dim,
                                        kernel_size))
    if bias:
      self.b1 = nn.Parameter(torch.Tensor(sym_groupings[1][0]*joint_out_dim))
      self.b2 = nn.Parameter(torch.Tensor(sym_groupings[1][-1]*joint_out_dim))
    else:
      self.register_parameter('b1', None)
      self.register_parameter('b2', None)
    self.reset_parameters()

  def reset_parameters(self):
    """Initializes the model parameters.

    Follows pytorch's default initialization of he Uniform for Conv1d layers.
    """
    # Initialize a big weights.
    ww_init = nn.Parameter(torch.Tensor(
        self.out_channels, self.in_channels, self.kernel_size))
    fan = nn.init._calculate_correct_fan(ww_init, 'fan_in')
    gain = nn.init.calculate_gain('leaky_relu', math.sqrt(5))
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    # Copy over part of the weights for symmetry.
    with torch.no_grad():
      nn.init.uniform_(self.w1, -bound, bound)
      nn.init.uniform_(self.w2, -bound, bound)
      nn.init.uniform_(self.w3, -bound, bound)
      nn.init.uniform_(self.w4, -bound, bound)
      nn.init.uniform_(self.w5, -bound, bound)

      if self.bias:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(ww_init)
        bound = 1. / math.sqrt(fan_in)
        nn.init.uniform_(self.b1, -bound, bound)
        nn.init.uniform_(self.b2, -bound, bound)

  def get_weight_(self,):
    """Constructs the full weight and bias with symmetry.

    Returns:
      ret (Tensor): Weight matrix for conv1d.
      ret_bias (Tensor): Bias vector for conv1d.
    """
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
    """Performs a conv1d forward pass.

    Args:
      x (Tensor): Input tensor follows pytorch's Conv1d convention.

    Returns:
     (Tensor): Output tensor follows pytorch's Conv1D convention.
    """
    big_weight, big_bias = self.get_weight_()
    return F.conv1d(x, big_weight, big_bias, self.stride,
                    self.padding, self.dilation, self.groups)
