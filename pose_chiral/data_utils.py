"""Implements utils for data processing."""


def get_group_lists(self, sym_grouping):
  """Gets the index list for left and right groups."""
  left_idx = [k for k in range(sym_grouping[0])]
  right_list = [k + sym_grouping[0] for k in range(sym_grouping[1])]
  return left_idx, right_list


def chiral_transform(x, num_joints, in_dim, neg_dim_in, sym_grouping):
  """Performs the chiral transformation.

  Args:
    x (Tensor): Input tensor of dimension [batch x joints x channels x time].
    num_joints: Number of joints.
    in_dim: Input dimension per joint.
    neg_dim_in: Number of dimensions needed to be negated.
    sym_groupings (tuple): Tuple consisting of in/out symmetry groupings.

  Returns:
    x (Tensor): Chiral pair of x, dimension [batch x joints x channels x time].
  """
  # Some checks.
  assert(num_joints == sum(sym_grouping))
  assert(x.shape[1] == num_joints)
  assert(x.shape[2] == in_dim)
  assert(neg_dim_in < in_dim)

  x_out = x.clone()
  # Negate the neg_dimensions.
  x_out[:, :, :neg_dim_in] *= -1

  # Swap the joint between left and right groups.
  left_idx, right_idx = get_group_lists(sym_grouping, sym_grouping)
  x_out[:, left_idx+right_idx] = x_out[:, right_idx+left_idx]
  return x_out
