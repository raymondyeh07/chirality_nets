"""Implements base class for chiral layers."""

import collections
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChiralLayerBase(nn.Module):
  """Base layer for symmetric grouping computation."""

  def __init__(self, in_channels, out_channels,
               sym_groupings=([6, 6, 5], [6, 6, 5]),
               neg_dim_in=1, neg_dim_out=1):
    """Base class initialization for chical layers.

    Input and output channel dimensions are assumed to be stacked in order of
    sym_groupings, e.g., [x_{left, wrist},...,
                          x_{right, wrist},...,
                          x_{center, hip},...].

    Each joint, for example x_{left, wrist}, is a stack of coordinates, where
    the coordinates that needed to be negated are stacked first. The number of
    negated dimensions are indicated by neg_dim_in. Simiarly, for the output
    dimensions.

    Note: The ordering is different from the notation used in the paper, due to
    notation convenience.

    Args:
      in_channels (int): Number of input channels.
      out_channels (int): Number of output channels.
      sym_groupings (tuple): Tuple consisting of symmetry groupings.
      neg_dim_in (int): Input dimension for `negated coordiantes'
      neg_dim_out (int): Output dimension for `negated coordiantes'
    """
    super(ChiralLayerBase, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.sym_groupings = sym_groupings
    self.neg_dim_in = neg_dim_in
    self.neg_dim_out = neg_dim_out

    self.num_joints_in = sum(self.sym_groupings[0])
    self.num_joints_out = sum(self.sym_groupings[1])

    # Checks if the sym_grouping for left and right have the same dimension.
    assert(self.sym_groupings[0][0] == self.sym_groupings[0][1])
    assert(self.sym_groupings[1][0] == self.sym_groupings[1][1])

    # Checks if channel number is a multiple of number of joints.
    assert(self.in_channels % self.num_joints_in == 0)
    assert(self.out_channels % self.num_joints_out == 0)

    self.joint_in_dim = self.in_channels // self.num_joints_in
    self.joint_out_dim = self.out_channels // self.num_joints_out

    # Get symmetric groupings index.
    sg_in = [0]
    sg_out = [0]
    for ng in self.sym_groupings[0]:
      sg_in.append(ng*self.joint_in_dim+sg_in[-1])
    for ng in self.sym_groupings[1]:
      sg_out.append(ng*self.joint_out_dim+sg_out[-1])

    # Get index for x_in and x_out.
    x_in_select = self._get_x_index(
        self.in_channels, neg_dim_in, self.joint_in_dim)
    x_out_select = self._get_x_index(
        self.out_channels, neg_dim_out, self.joint_out_dim)

    # Get index for yz_in and yz_out.
    yz_in_select = list(set(range(self.in_channels))-set(x_in_select))
    yz_out_select = list(set(range(self.out_channels))-set(x_out_select))

    self.neg_in_select = x_in_select
    self.neg_out_select = x_out_select
    self.pos_in_select = yz_in_select
    self.pos_out_select = yz_out_select

    # Get additional index.
    x_select_group = collections.defaultdict(list)
    yz_select_group = collections.defaultdict(list)

    # Get x,yz select grouping index.
    for ind, ng in enumerate(self.sym_groupings[0]):
      x_select_group['in'].append(torch.tensor(
          [xx for xx in x_in_select if xx < sg_in[ind+1] and xx >= sg_in[ind]]).view(-1, 1))
      yz_select_group['in'].append(torch.tensor(
          [xx for xx in yz_in_select if xx < sg_in[ind+1] and xx >= sg_in[ind]]).view(-1, 1))
    for ind, ng in enumerate(self.sym_groupings[1]):
      x_select_group['out'].append(torch.tensor(
          [xx for xx in x_out_select if xx < sg_out[ind+1] and xx >= sg_out[ind]]).view(-1, 1))
      yz_select_group['out'].append(torch.tensor(
          [xx for xx in yz_out_select if xx < sg_out[ind+1] and xx >= sg_out[ind]]).view(-1, 1))

    self.neg_select_group = x_select_group
    self.pos_select_group = yz_select_group

  def _get_x_index(self, num_channel, x_dim, joint_dim):
    """Returns the index for the 'negated coordinates' of each of the joints.

    Args:
      num_channels (int): Number of channels.
      x_dim (int): Number of negated dimensions.

    Returns:
      x_select (list): List of negated dimension index.
    """
    x_select = [k for k in range(0, num_channel, joint_dim)]
    x_select = [k+j for k in x_select for j in range(x_dim)]
    return x_select
