"""Test for Chiral conv1d layers."""
import unittest

import torch
import torch.nn.functional as F

import sys
sys.path.append("pose_chiral/")

import data_utils


class TestChiralBase(unittest.TestCase):
  """Implements unittests for chiral conv1d layers."""

  def setUp(self,):
    pass

  def _get_input_pairs(self, batch_size=2, time_size=5, num_joints=3,
                       in_dim=2,
                       neg_dim_in=1,
                       sym_groupings=([1, 1, 1], [1, 1, 1])):
    """Returns a pair of chiral inputs.

    Args:
      batch_size (int): Number of example per batch.
      time_size (int): Number of time step per example.
      num_joints (int): Number of joints in the input.
      in_dim (int): Input dimension per joint.
      neg_dim_in (int): Number of dimension needed to be negated.
      sym_groupings (tuple): Tuple consisting of in/out symmetry groupings.

    Returns:
      x (Tensor): Tensor of dimension [batch x joints x channels x time].
      x_chiral (Tensor): Chiral pair of x.
    """
    x = torch.randn(batch_size, num_joints, in_dim, time_size)
    x_chiral = data_utils.chiral_transform(x, num_joints,
                                           in_dim, neg_dim_in,
                                           sym_groupings[0])
    return x, x_chiral

  def _checks_chiral_equivariant(self, x, x_chiral, num_joints,
                                 in_dim, neg_dim_in, sym_grouping):
    """Checks the chiral equivariance is satisfied, diff < 1e-8."""
    x_chiral_chiral = data_utils.chiral_transform(x_chiral, num_joints,
                                                  in_dim, neg_dim_in,
                                                  sym_grouping)
    diff_val = (x-x_chiral_chiral)**2
    diff_sum = diff_val.sum()
    print('Difference from equivariance: %s' % diff_sum.data.cpu().numpy())
    print('')
    assert(diff_sum < 1e-8)
