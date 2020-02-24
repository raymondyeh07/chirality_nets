"""Test for Chiral RNN layers, including stacked LSTM and GRU."""
import unittest

import torch
import torch.nn.functional as F

from tests.test_chiral_base import TestChiralBase
from chiral_layers.chiral_lstm import ChiralLstm
from chiral_layers.chiral_gru import ChiralGru


class TestChiralRnns(TestChiralBase):
  """Implements unittests for chiral conv1d layers."""

  def test_lstm_equi_group(self):
    """Performs unittest for lstm equivariance."""
    print('Tests equivariance of LSTM.')
    batch_size = 2
    time_size = 1
    num_joints = 5
    in_dim = 2
    out_dim = 2
    neg_dim_in = 1
    neg_dim_out = 1
    sym_groupings = ([2, 2, 1], [2, 2, 1])

    # Generate chiral pairs.
    x, x_chiral = self._get_input_pairs(batch_size, time_size, num_joints,
                                        in_dim, neg_dim_in, sym_groupings)
    # Permute to time in first index.
    x = x.permute(-1, 0, 1, 2)
    x_chiral = x_chiral.permute(-1, 0, 1, 2)
    # Reshape to time x batch x channels.
    x = x.view(x.shape[0], x.shape[1], -1)
    x_chiral = x_chiral.view(x.shape[0], x.shape[1], -1)

    chiral_model = ChiralLstm(num_joints*in_dim, num_joints*in_dim,
                              num_layers=2, bias=True,
                              dropout=0.,
                              sym_groupings=sym_groupings,
                              neg_dim_in=neg_dim_in, neg_dim_out=neg_dim_out)
    y, _ = chiral_model(x)
    y_chiral, _ = chiral_model(x_chiral)

    # Reshape back to joints, dim representation.
    y = y.view(y.shape[0], batch_size, num_joints, -1)
    y_chiral = y_chiral.view(y_chiral.shape[0], batch_size, num_joints, -1)

    # Permute time back to last dimension.
    y = y.permute(1, 2, 3, 0)
    y_chiral = y_chiral.permute(1, 2, 3, 0)
    # Compare output.
    self._checks_chiral_equivariant(y, y_chiral, num_joints, out_dim,
                                    neg_dim_out, sym_groupings[1])

  def test_gru_equi_group(self):
    """Performs unittest for gru equivariance."""
    print('Tests equivariance of GRU.')
    batch_size = 2
    time_size = 1
    num_joints = 5
    in_dim = 2
    out_dim = 2
    neg_dim_in = 1
    neg_dim_out = 1
    sym_groupings = ([2, 2, 1], [2, 2, 1])

    # Generate chiral pairs.
    x, x_chiral = self._get_input_pairs(batch_size, time_size, num_joints,
                                        in_dim, neg_dim_in, sym_groupings)
    # Permute to time in first index.
    x = x.permute(-1, 0, 1, 2)
    x_chiral = x_chiral.permute(-1, 0, 1, 2)
    # Reshape to time x batch x channels.
    x = x.view(x.shape[0], x.shape[1], -1)
    x_chiral = x_chiral.view(x.shape[0], x.shape[1], -1)

    chiral_model = ChiralGru(num_joints*in_dim, num_joints*in_dim,
                             num_layers=2, bias=True,
                             dropout=0.,
                             sym_groupings=sym_groupings,
                             neg_dim_in=neg_dim_in, neg_dim_out=neg_dim_out)
    y, _ = chiral_model(x)
    y_chiral, _ = chiral_model(x_chiral)

    # Reshape back to joints, dim representation.
    y = y.view(y.shape[0], batch_size, num_joints, -1)
    y_chiral = y_chiral.view(y_chiral.shape[0], batch_size, num_joints, -1)

    # Permute time back to last dimension.
    y = y.permute(1, 2, 3, 0)
    y_chiral = y_chiral.permute(1, 2, 3, 0)
    # Compare output.
    self._checks_chiral_equivariant(y, y_chiral, num_joints, out_dim,
                                    neg_dim_out, sym_groupings[1])


if __name__ == '__main__':
  unittest.main()
