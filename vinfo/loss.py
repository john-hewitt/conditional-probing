"""Custom loss classes for probing tasks."""

import torch
import torch.nn as nn
from tqdm import tqdm

class CustomCrossEntropyLoss(nn.Module):
  """Custom cross-entropy loss"""
  def __init__(self, args):
    super(CustomCrossEntropyLoss, self).__init__()
    tqdm.write('Constructing CrossEntropyLoss')
    self.args = args
    self.pytorch_ce_loss = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    self.to(args['device'])

  def forward(self, predictions, label_batch):
    """
    Computes and returns CrossEntropyLoss.

    Ignores all entries where label_batch=-1
    Noralizes by the number of sentences in the batch.

    Args: 
      predictions: A pytorch batch of logits
      label_batch: A pytorch batch of label indices
      length_batch: A pytorch batch of sentence lengths

    Returns:
      A tuple of:
        cross_entropy_loss: average loss in the batch
        total_sents: number of sentences in the batch
    """
    if len(predictions.shape) == 3:
      batchlen, seqlen, class_count = predictions.shape
      predictions = predictions.view(batchlen*seqlen, class_count)
      label_batch = label_batch.view(batchlen*seqlen).long()
      cross_entropy_loss = self.pytorch_ce_loss(predictions, label_batch)
      count = torch.sum((label_batch != 0).long())
    elif len(predictions.shape) == 2:
      batchlen, class_count = predictions.shape
      predictions = predictions.view(batchlen, class_count)
      label_batch = label_batch.view(batchlen).long()
      #print(predictions)
      #print(label_batch)
      cross_entropy_loss = self.pytorch_ce_loss(predictions, label_batch)
      count = torch.sum((label_batch != 0).long())
    return cross_entropy_loss, count
