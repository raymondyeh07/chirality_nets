"""Test for chiral batch_norm1d layer."""
import unittest

import torch
import torch.nn.functional as F

from tests.test_chiral_base import TestChiralBase
from chiral_layers.chiral_batch_norm1d import ChiralBatchNorm1d


class TestChiralBatchNorm1d(TestChiralBase):
  """Implements unittests for chiral conv1d layers."""

  def test_single_layer_running_mean_var_updates(self):
    print('Tests batchnorm running mean and var updates.')
    batch_size = 2
    time_size = 5
    num_joints = 5
    in_dim = 2
    out_dim = 2
    neg_dim_in = 1
    neg_dim_out = 1
    sym_groupings = ([2, 2, 1], [2, 2, 1])

    # Generate chiral pairs.
    x, x_chiral = self._get_input_pairs(batch_size, time_size, num_joints,
                                        in_dim, neg_dim_in, sym_groupings)

    # Reshape to joints and dim to a column vector.
    x = x.view(x.shape[0], -1, x.shape[-1])
    x_chiral = x_chiral.view(x.shape[0], -1, x.shape[-1])

    bn_layer = ChiralBatchNorm1d(num_joints*in_dim,
                                 sym_groupings=sym_groupings,
                                 neg_dim_in=neg_dim_in,
                                 neg_dim_out=neg_dim_out)
    # Checks mean is initialize
    bn_mean = bn_layer.running_mean_neg1.mean()
    assert(torch.eq(bn_mean, torch.zeros_like(bn_mean)))

    # Checks variance is initialized.
    bn_var = bn_layer.running_var_neg1.mean()
    assert(torch.eq(bn_var, torch.ones_like(bn_var)))

    bn_layer.train()
    for k in range(5):
      xx, _ = self._get_input_pairs(batch_size, time_size, num_joints,
                                    in_dim, neg_dim_in, sym_groupings)
      bn_layer(x)

    # Checks mean is updated, thus not zero.
    bn_mean = bn_layer.running_mean_neg1.mean()
    assert(not torch.eq(bn_mean, torch.zeros_like(bn_mean)))

    # Checks variance is updated, thus not one.
    bn_var = bn_layer.running_var_neg1.mean()
    assert(not torch.eq(bn_var, torch.ones_like(bn_var)))

  def test_single_layer_equi_at_test(self):
    """Performs unittest for equivariance on toy example"""
    print('Tests batchnorm equivariance at test time.')
    batch_size = 2
    time_size = 5
    num_joints = 5
    in_dim = 2
    out_dim = 2
    neg_dim_in = 1
    neg_dim_out = 1
    sym_groupings = ([2, 2, 1], [2, 2, 1])

    # Generate chiral pairs.
    x, x_chiral = self._get_input_pairs(batch_size, time_size, num_joints,
                                        in_dim, neg_dim_in, sym_groupings)

    # Reshape to joints and dim to a column vector.
    x = x.view(x.shape[0], -1, x.shape[-1])
    x_chiral = x_chiral.view(x.shape[0], -1, x.shape[-1])

    bn_layer = ChiralBatchNorm1d(num_joints*in_dim,
                                 sym_groupings=sym_groupings,
                                 neg_dim_in=neg_dim_in,
                                 neg_dim_out=neg_dim_out)

    # Perform training to update running stats.
    bn_layer.train()
    for k in range(5):
      xx, _ = self._get_input_pairs(batch_size, time_size, num_joints,
                                    in_dim, neg_dim_in, sym_groupings)
      bn_layer(x)

    # Perform forward pass at test time.
    bn_layer.eval()
    y = bn_layer(x)
    y_chiral = bn_layer(x_chiral)

    # Reshape back to joints, dim representation.
    y = y.view(y.shape[0], num_joints, -1, y.shape[-1])
    y_chiral = y_chiral.view(y.shape[0], num_joints, -1, y.shape[-1])

    # Compare output.
    self._checks_chiral_equivariant(y, y_chiral, num_joints, out_dim,
                                    neg_dim_out, sym_groupings[1])


if __name__ == '__main__':
  unittest.main()
