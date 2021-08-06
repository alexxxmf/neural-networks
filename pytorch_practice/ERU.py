import torch.nn as nn
import torch


def eru(input: torch.Tensor, *, radix:float=1.) -> torch.Tensor:
  i_radix = 1/radix
  s_radix = radix ** radix

  negative_case_tensor = torch.exp(torch.mul(input, radix)) - i_radix
  positive_or_zero_case_tensor = torch.pow((torch.mul(input, s_radix) + 1), i_radix) - i_radix

  return torch.where(input < 0,
                    negative_case_tensor,
                    positive_or_zero_case_tensor)

class ERU(nn.Module):
  '''
  Applies the Exponential Root Unit Unit (ERU) function element-wise.
  Args:
    radix: is a positive shape parameter that defines the choice of non-linearity
  Shape:
    - Input: (N, *) where * means, any number of additional
      dimensions
    - Output: (N, *), same shape as the input

  References:
    -  Related paper:
    https://arxiv.org/pdf/1804.11237.pdf

  Examples:
    >>> e1ru = ERU(radix=1)
    >>> input = torch.randn(2)
    >>> output = e1ru(input)
  '''
  def __init__(self, radix:float=1.):
    super().__init__()
    self.radix = radix

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    return eru(input, radix=self.radix)