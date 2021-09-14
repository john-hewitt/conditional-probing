"""Classes for specifying probe pytorch modules.
Draws from https://github.com/john-hewitt/structural-probes"""

import torch.nn as nn
import torch
import numpy
from tqdm import tqdm
from yaml import YAMLObject

from utils import InitYAMLObject

class Probe(nn.Module, InitYAMLObject):

  def print_param_count(self):
    total_params = 0
    for param in self.parameters():
      total_params += numpy.prod(param.size())
    tqdm.write('Probe has {} parameters'.format(total_params))

class OneWordLinearLabelProbe(Probe):
  """ Computes a linear function of each word vector.

  For a batch of sentences, computes all n scores
  for each sentence in the batch.
  """
  yaml_tag = '!OneWordLinearLabelProbe'
  def __init__(self, args, model_dim, label_space_size, zero_features=False):
    print('Constructing OneWordLinearLabelProbe')
    super(OneWordLinearLabelProbe, self).__init__()
    self.args = args
    self.model_dim = model_dim
    self.label_space_size = label_space_size
    self.linear = nn.Linear(self.model_dim, self.label_space_size)
    #self.linear2 = nn.Linear(self.label_space_size, self.label_space_size)
    self.print_param_count()
    dropout = .0
    self.dropout = nn.Dropout(p=dropout)
    print('Applying dropout {}'.format(dropout))
    self.zero_features = zero_features
    self.to(args['device'])

  def forward(self, batch):
    """ Computes all n label logits for each sentence in a batch.

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of logits of shape (batch_size, max_seq_len)
    """
    #batchlen, seqlen, dimension = batch.size()
    #batch = self.dropout(batch)
    #batch = self.linear1(batch)
    #logits = self.linear2(batch)
    if self.zero_features:
      batch = torch.zeros_like(batch)
    batch = self.linear(batch)
    #batch = self.linear2(torch.nn.functional.gelu(batch))
    return batch

class OneWordMLPLabelProbe(Probe):
  """ Computes a linear function of each word vector.

  For a batch of sentences, computes all n scores
  for each sentence in the batch.
  """
  yaml_tag = '!OneWordMLPLabelProbe'
  def __init__(self, args, model_dim, label_space_size, zero_features=False):
    print('Constructing OneWordLinearLabelProbe')
    super(OneWordMLPLabelProbe, self).__init__()
    self.args = args
    self.model_dim = model_dim
    self.label_space_size = label_space_size
    self.linear = nn.Linear(self.model_dim, 100)
    self.linear2 = nn.Linear(100, self.label_space_size)
    self.print_param_count()
    dropout = .0
    self.dropout = nn.Dropout(p=dropout)
    print('Applying dropout {}'.format(dropout))
    self.zero_features = zero_features
    self.to(args['device'])

  def forward(self, batch):
    """ Computes all n label logits for each sentence in a batch.

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of logits of shape (batch_size, max_seq_len)
    """
    #batchlen, seqlen, dimension = batch.size()
    #batch = self.dropout(batch)
    #batch = self.linear1(batch)
    #logits = self.linear2(batch)
    if self.zero_features:
      batch = torch.zeros_like(batch)
    batch = self.linear(batch)
    batch = self.linear2(torch.nn.functional.gelu(batch))
    return batch

class SentenceLinearLabelProbe(Probe):
  """ Computes a linear function of pairs of vectors.

  For a batch of sentences, computes all n scores
  for each sentence in the batch.
  """
  yaml_tag = '!SentenceLinearLabelProbe'
  def __init__(self, args, model_dim, label_space_size, zero_features=False):
    print('Constructing SentenceLinearLabelProbe')
    super(SentenceLinearLabelProbe, self).__init__()
    self.args = args
    self.model_dim = model_dim
    self.label_space_size = label_space_size
    self.linear = nn.Linear(self.model_dim, self.label_space_size)
    self.linear2 = nn.Linear(self.label_space_size, self.label_space_size)
    self.print_param_count()
    dropout = 0
    self.dropout = nn.Dropout(p=0)
    print('Applying dropout {}'.format(dropout))
    self.zero_features = zero_features
    self.to(args['device'])

  def forward(self, batch):
    """ Computes all n label logits for each sentence in a batch.

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of logits of shape (batch_size, max_seq_len)
    """
    #batch = torch.max(batch, dim=1).values
    batch = torch.mean(batch, dim=1)
    if self.zero_features:
      batch = torch.zeros_like(batch)
    #batch = self.linear2(torch.nn.functional.gelu(self.linear(batch)))
    batch = self.linear(batch)
    return batch


class SentenceMLPLabelProbe(Probe):
  """ Computes a linear function of pairs of vectors.

  For a batch of sentences, computes all n scores
  for each sentence in the batch.
  """
  yaml_tag = '!SentenceMLPLabelProbe'
  def __init__(self, args, model_dim, label_space_size, zero_features=False):
    print('Constructing SentenceLinearLabelProbe')
    super(SentenceMLPLabelProbe, self).__init__()
    self.args = args
    self.model_dim = model_dim
    self.label_space_size = label_space_size
    self.linear = nn.Linear(self.model_dim, 100)
    self.linear2 = nn.Linear(100, self.label_space_size)
    self.print_param_count()
    dropout = 0
    self.dropout = nn.Dropout(p=0)
    print('Applying dropout {}'.format(dropout))
    self.zero_features = zero_features
    self.to(args['device'])

  def forward(self, batch):
    """ Computes all n label logits for each sentence in a batch.

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of logits of shape (batch_size, max_seq_len)
    """
    #batch = torch.max(batch, dim=1).values
    batch = torch.mean(batch, dim=1)
    if self.zero_features:
      batch = torch.zeros_like(batch)
    batch = self.linear2(torch.nn.functional.gelu(self.linear(batch)))
    return batch
