"""Test for Chiral conv1d layers."""
import unittest

import torch
import torch.nn.functional as F

from tests.test_chiral_base import TestChiralBase
from chiral_layers.chiral_conv1d import ChiralConv1d


class TestChiralConv1d(TestChiralBase):
  """Implements unittests for chiral conv1d layers."""

  def test_single_layer_equi_group1(self):
    """Performs unittest for equivariance on toy example"""
    print('Test equivariance for conv1d layer, sym_group [1,1,1].')
    batch_size = 2
    time_size = 5
    num_joints = 3
    in_dim = 2
    out_dim = 3
    neg_dim_in = 1
    neg_dim_out = 1
    sym_groupings = ([1, 1, 1], [1, 1, 1])

    # Generate chiral pairs.
    x, x_chiral = self._get_input_pairs(batch_size, time_size, num_joints,
                                        in_dim, neg_dim_in, sym_groupings)

    # Reshape to joints and dim to a column vector.
    x = x.view(x.shape[0], -1, x.shape[-1])
    x_chiral = x_chiral.view(x.shape[0], -1, x.shape[-1])

    chiral_conv = ChiralConv1d(in_dim*num_joints, out_dim*num_joints,
                               kernel_size=3,
                               sym_groupings=sym_groupings,
                               neg_dim_in=neg_dim_in,
                               neg_dim_out=neg_dim_out)
    y = chiral_conv(x)
    y_chiral = chiral_conv(x_chiral)

    # Reshape back to joints, dim representation.
    y = y.view(y.shape[0], num_joints, -1, y.shape[-1])
    y_chiral = y_chiral.view(y.shape[0], num_joints, -1, y.shape[-1])

    # Compare output.
    self._checks_chiral_equivariant(y, y_chiral, num_joints, out_dim,
                                    neg_dim_out, sym_groupings[1])

  def test_single_layer_equi_group2(self):
    """Performs unittest for equivariance with different sym groups."""
    print('Test equivariance for conv1d layer, sym_group [2,2,1].')
    batch_size = 2
    time_size = 5
    num_joints = 5
    in_dim = 2
    out_dim = 3
    neg_dim_in = 1
    neg_dim_out = 1
    sym_groupings = ([2, 2, 1], [2, 2, 1])

    # Generate chiral pairs.
    x, x_chiral = self._get_input_pairs(batch_size, time_size, num_joints,
                                        in_dim, neg_dim_in, sym_groupings)

    # Reshape to joints and dim to a column vector.
    x = x.view(x.shape[0], -1, x.shape[-1])
    x_chiral = x_chiral.view(x.shape[0], -1, x.shape[-1])

    chiral_conv = ChiralConv1d(in_dim*num_joints, out_dim*num_joints,
                               kernel_size=3,
                               sym_groupings=sym_groupings,
                               neg_dim_in=neg_dim_in,
                               neg_dim_out=neg_dim_out)
    y = chiral_conv(x)
    y_chiral = chiral_conv(x_chiral)

    # Reshape back to joints, dim representation.
    y = y.view(y.shape[0], num_joints, -1, y.shape[-1])
    y_chiral = y_chiral.view(y.shape[0], num_joints, -1, y.shape[-1])

    # Compare output.
    self._checks_chiral_equivariant(y, y_chiral, num_joints, out_dim,
                                    neg_dim_out, sym_groupings[1])

  def test_single_layer_equi_diff_in_out_joints(self):
    """Performs unittest for equivariance for different in/out joints."""
    print('Test equivariance for conv1d layer, different in/out sym_group.')
    batch_size = 2
    time_size = 5
    num_joints_in = 5
    num_joints_out = 3
    in_dim = 2
    out_dim = 3
    neg_dim_in = 1
    neg_dim_out = 1
    sym_groupings = ([2, 2, 1], [1, 1, 1])

    # Generate chiral pairs.
    x, x_chiral = self._get_input_pairs(batch_size, time_size, num_joints_in,
                                        in_dim, neg_dim_in, sym_groupings)

    # Reshape to joints and dim to a column vector.
    x = x.view(x.shape[0], -1, x.shape[-1])
    x_chiral = x_chiral.view(x.shape[0], -1, x.shape[-1])

    chiral_conv = ChiralConv1d(in_dim*num_joints_in, out_dim*num_joints_out,
                               kernel_size=3,
                               sym_groupings=sym_groupings,
                               neg_dim_in=neg_dim_in,
                               neg_dim_out=neg_dim_out)
    y = chiral_conv(x)
    y_chiral = chiral_conv(x_chiral)

    # Reshape back to joints, dim representation.
    y = y.view(y.shape[0], num_joints_out, -1, y.shape[-1])
    y_chiral = y_chiral.view(y.shape[0], num_joints_out, -1, y.shape[-1])

    # Compare output.
    self._checks_chiral_equivariant(y, y_chiral, num_joints_out, out_dim,
                                    neg_dim_out, sym_groupings[1])


if __name__ == '__main__':
  unittest.main()
