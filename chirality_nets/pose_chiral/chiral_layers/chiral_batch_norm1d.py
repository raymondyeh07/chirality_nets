"""Implements a chiral batch_norm1d layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from chiral_layers.chiral_layer_base import ChiralLayerBase


class ChiralBatchNorm1d(ChiralLayerBase):
  """Implements batch normalization 1d with chiral equivariance."""

  def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
               track_running_stats=True, sym_groupings=([6, 6, 5], [6, 6, 5]),
               neg_dim_in=1, neg_dim_out=1):
    """Applies Batch Normalization with chiral equivariance.

    Args:
      num_features (int): Number of channels in the input.
      eps (float): a value added to the denominator for numerical stability.
      momentum (float): the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
      affine (bool): a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
      track_running_stats (bool): a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``
      sym_groupings (tuple): Tuple consisting of in/out symmetry groupings.
      neg_dim_in (int): Input dimension for `negated coordiantes'.
      neg_dim_out (int): Output dimension for `negated coordiantes'.
    """
    # Input and output dimension and symmetry are the same for batchnorm.
    assert(neg_dim_in == neg_dim_in)
    assert(sym_groupings[0] == sym_groupings[1])
    super(ChiralBatchNorm1d, self).__init__(num_features,
                                            num_features, sym_groupings,
                                            neg_dim_in, neg_dim_out)
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    self.affine = affine
    self.track_running_stats = track_running_stats

    self.neg_in_len_sym = len(self.neg_select_group['out'][0])
    self.pos_in_len_sym = len(self.pos_select_group['out'][0])
    self.neg_in_len_center = len(self.neg_select_group['out'][2])
    self.pos_in_len_center = len(self.pos_select_group['out'][2])

    if affine:
      self.bn_neg_w1 = nn.Parameter(torch.Tensor(self.neg_in_len_sym))
      self.bn_neg_b1 = nn.Parameter(torch.Tensor(self.neg_in_len_sym))
      self.bn_neg_w2 = nn.Parameter(torch.Tensor(self.neg_in_len_center))
      self.bn_neg_b2 = nn.Parameter(torch.Tensor(self.neg_in_len_center))

      self.bn_pos_w1 = nn.Parameter(torch.Tensor(self.pos_in_len_sym))
      self.bn_pos_b1 = nn.Parameter(torch.Tensor(self.pos_in_len_sym))
      self.bn_pos_w2 = nn.Parameter(torch.Tensor(self.pos_in_len_center))
      self.bn_pos_b2 = nn.Parameter(torch.Tensor(self.pos_in_len_center))
    else:
      for k in range(1, 3):
        self.register_parameter('bn_neg_w%s' % k, None)
        self.register_parameter('bn_neg_b%s' % k, None)
        self.register_parameter('bn_pos_w%s' % k, None)
        self.register_parameter('bn_pos_b%s' % k, None)

    if track_running_stats:
      for k in range(1, 3):
        if k == 1:
          llx = self.neg_in_len_sym
          llpos = self.pos_in_len_sym
        else:
          llx = self.neg_in_len_center
          llpos = self.pos_in_len_center
        self.register_buffer('running_mean_neg%s' % k,
                             torch.zeros(llx))
        self.register_buffer('running_var_neg%s' % k,
                             torch.ones(llx))
        self.register_buffer('running_mean_pos%s' % k,
                             torch.zeros(llpos))
        self.register_buffer('running_var_pos%s' % k,
                             torch.ones(llpos))
        self.register_buffer('num_batches_tracked',
                             torch.tensor(0, dtype=torch.long))
    else:
      for k in range(1, 3):
        self.register_parameter('running_mean_neg%s' % k, None)
        self.register_parameter('running_var_neg%s' % k, None)
        self.register_parameter('running_mean_pos%s' % k, None)
        self.register_parameter('running_var_pos%s' % k, None)
      self.register_parameter('num_batches_tracked', None)

    self.reset_parameters()

  def set_bn_momentum(self, momentum):
    self.momentum = momentum

  def _running_update(self, running_stat, new_stat):
    return (1.-self.momentum)*running_stat + self.momentum*new_stat

  def _update_running_stats(self, neg1, neg2, neg3, pos1, pos2, pos3):
    with torch.no_grad():
      # Compute mean per group.
      neg1_per, neg1_mean = self._get_permute_mean(neg1)
      neg2_per, neg2_mean = self._get_permute_mean(neg2)
      neg3_per, neg3_mean = self._get_permute_mean(neg3)
      pos1_per, pos1_mean = self._get_permute_mean(pos1)
      pos2_per, pos2_mean = self._get_permute_mean(pos2)
      pos3_per, pos3_mean = self._get_permute_mean(pos3)

      # Compute overall mean and variance.
      neg12_mean = (neg1_mean - neg2_mean) / 2.  # Odd symmetry.
      neg12_var1 = ((neg1_per - neg12_mean.unsqueeze(-1))**2).mean(1)
      neg12_var2 = ((neg2_per + neg12_mean.unsqueeze(-1))**2).mean(1)
      neg12_var = (neg12_var1 + neg12_var2) / 2.

      pos12_mean = (pos1_mean + pos2_mean) / 2.  # Even symmetry.
      pos12_var1 = ((pos1_per - pos12_mean.unsqueeze(-1))**2).mean(1)
      pos12_var2 = ((pos2_per - pos12_mean.unsqueeze(-1))**2).mean(1)
      pos12_var = (pos12_var1 + pos12_var2) / 2.

      neg3_mean = neg3_mean * 0.  # Due to odd symmetry, mean is zero.
      neg3_var = ((neg3_per - neg3_mean.unsqueeze(-1))**2).mean(1)
      pos3_var = ((pos3_per - pos3_mean.unsqueeze(-1))**2).mean(1)

      # Update running mean and variance.
      self.running_mean_neg1 = self._running_update(
          self.running_mean_neg1, neg12_mean)
      self.running_mean_pos1 = self._running_update(
          self.running_mean_pos1, pos12_mean)
      self.running_mean_neg2 = self._running_update(
          self.running_mean_neg2, neg3_mean)
      self.running_mean_pos2 = self._running_update(
          self.running_mean_pos2, pos3_mean)
      self.running_var_neg1 = self._running_update(
          self.running_var_neg1, neg12_var)
      self.running_var_pos1 = self._running_update(
          self.running_var_pos1, pos12_var)
      self.running_var_neg2 = self._running_update(
          self.running_var_neg2, neg3_var)
      self.running_var_pos2 = self._running_update(
          self.running_var_pos2, pos3_var)

  def _get_permute_mean(self, x):
    """Permute and compute mean over batch and time.
    """
    x_per = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
    x_mean = x_per.mean(1)
    return x_per, x_mean

  def _get_mean_var(self, x):
    x_per, x_mean = self._get_permute_mean(x)
    x_var = ((x_per-x_mean.unsqueeze(-1))**2).mean(1)
    return x_mean, x_var

  def forward(self, x):
    out_idx = torch.cat([self.neg_select_group['out'][0],
                         self.pos_select_group['out'][0],
                         self.neg_select_group['out'][1],
                         self.pos_select_group['out'][1],
                         self.neg_select_group['out'][2],
                         self.pos_select_group['out'][2]]).unsqueeze(-1)
    if self.training:
      bn_mean, bn_var = self._get_mean_var(x)
      # Split input up into groups, and compute batch mean and var.
      neg1 = x[:, self.neg_select_group['out'][0].squeeze(-1), :]
      neg2 = x[:, self.neg_select_group['out'][1].squeeze(-1), :]
      neg3 = x[:, self.neg_select_group['out'][2].squeeze(-1), :]
      pos1 = x[:, self.pos_select_group['out'][0].squeeze(-1), :]
      pos2 = x[:, self.pos_select_group['out'][1].squeeze(-1), :]
      pos3 = x[:, self.pos_select_group['out'][2].squeeze(-1), :]
      # If training update running mean.
      self._update_running_stats(neg1, neg2, neg3, pos1, pos2, pos3)
    else:
      # During eval use running mean and var.
      bn_mean_tmp = torch.cat([self.running_mean_neg1,
                               self.running_mean_pos1,
                               -self.running_mean_neg1,
                               self.running_mean_pos1,
                               0.*self.running_mean_neg2,
                               self.running_mean_pos2])
      bn_var_tmp = torch.cat([self.running_var_neg1, self.running_var_pos1,
                              self.running_var_neg1, self.running_var_pos1,
                              self.running_var_neg2, self.running_var_pos2
                              ])
      bn_mean = torch.zeros_like(bn_mean_tmp)
      bn_var = torch.zeros_like(bn_mean_tmp)
      bn_mean[out_idx.squeeze()] = bn_mean_tmp
      bn_var[out_idx.squeeze()] = bn_var_tmp

    bn_mean = bn_mean.unsqueeze(0).unsqueeze(-1)
    bn_var = bn_var.unsqueeze(0).unsqueeze(-1)
    bn_w_tmp = torch.cat([self.bn_neg_w1, self.bn_pos_w1,
                          self.bn_neg_w1, self.bn_pos_w1,
                          self.bn_neg_w2, self.bn_pos_w2])
    bn_b_tmp = torch.cat([self.bn_neg_b1, self.bn_pos_b1,
                          -self.bn_neg_b1, self.bn_pos_b1,
                          0.*self.bn_neg_b2, self.bn_pos_b2])
    # Assign to output ordering.
    bn_w = torch.zeros_like(bn_w_tmp)
    bn_b = torch.zeros_like(bn_b_tmp)
    bn_w[out_idx.squeeze()] = bn_w_tmp
    bn_b[out_idx.squeeze()] = bn_b_tmp
    bn_w = bn_w.unsqueeze(0).unsqueeze(-1)
    bn_b = bn_b.unsqueeze(0).unsqueeze(-1)
    # Perform bn forward pass
    ret = (x - bn_mean) / ((self.eps + bn_var)**0.5) * bn_w + bn_b
    return ret

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean_neg1.zero_()
      self.running_mean_neg2.zero_()
      self.running_var_neg1.fill_(1)
      self.running_var_neg2.fill_(1)
      self.running_mean_pos1.zero_()
      self.running_mean_pos2.zero_()
      self.running_var_pos1.fill_(1)
      self.running_var_pos2.fill_(1)

      self.num_batches_tracked.zero_()

  def reset_parameters(self):
    self.reset_running_stats()
    if self.affine:
      nn.init.uniform_(self.bn_neg_w1)
      nn.init.zeros_(self.bn_neg_b1)
      nn.init.uniform_(self.bn_neg_w2)
      nn.init.zeros_(self.bn_neg_b2)
      nn.init.uniform_(self.bn_pos_w1)
      nn.init.zeros_(self.bn_pos_b1)
      nn.init.uniform_(self.bn_pos_w2)
      nn.init.zeros_(self.bn_pos_b2)
