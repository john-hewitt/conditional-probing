from collections import defaultdict
import os
from yaml import YAMLObject
from utils import InitYAMLObject
from stanza.models.ner.scorer import score_by_entity

from tqdm import tqdm
#from scipy.stats import spearmanr, pearsonr
import numpy as np 
import json

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
mpl.rcParams['agg.path.chunksize'] = 10000

import torch

class Reporter(InitYAMLObject):
  """Base class for reporting.

  Attributes:
    test_reporting_constraint: Any reporting method
      (identified by a string) not in this list will not
      be reported on for the test set.
  """

  def __init__(self, args, dataset):
    raise NotImplementedError("Inherit from this class and override __init__")

  def __call__(self, prediction_batches, dataloader, split_name):
    """
    Performs all reporting methods as specifed in the yaml experiment config dict.
    
    Any reporting method not in test_reporting_constraint will not
      be reported on for the test set.

    Args:
      prediction_batches: A sequence of batches of predictions for a data split
      dataloader: A DataLoader for a data split
      split_name the string naming the data split: {train,dev,test}
    """
    for method in self.reporting_methods:
      if method in self.reporting_method_dict:
        if split_name == 'test' and method not in self.test_reporting_constraint:
          tqdm.write("Reporting method {} not in test set reporting "
              "methods (reporter.py); skipping".format(method))
          continue
        tqdm.write("Reporting {} on split {}".format(method, split_name))
        self.reporting_method_dict[method](prediction_batches
            , dataloader, split_name)
      else:
        tqdm.write('[WARNING] Reporting method not known: {}; skipping'.format(method))

class IndependentLabelReporter(Reporter):
  """
  Class for computing and reporting metrics on
  tasks where each label output of the prediction
  should be compared to its corresponding label,
  and accuracy should be computed by taking the
  percent of correct labels over all outputs
  (not including the pad label).

  This is as opposed to NER, for example, in which
  span-based evaluation is required.

  But works for PoS, dep, maybe even NLI; not sure yet
  """
  yaml_tag = '!IndependentLabelReporter'

  def __init__(self, args, reporting_root, reporting_methods):
    self.args = args
    self.reporting_methods = reporting_methods
    self.reporting_method_dict = {
        'label_accuracy':self.report_label_values,
        'v_entropy':self.report_v_entropy,
        }
    #self.reporting_root = args['reporting']['root']
    self.reporting_root = reporting_root
    self.test_reporting_constraint = {'label_accuracy', 'v_entropy'}


  def report_label_values(self, prediction_batches, dataset, split_name):
    total = 0
    correct = 0
    for prediction_batch, (_, label_batch, sentences) in zip(prediction_batches, dataset):
      prediction_batch = prediction_batch.to(self.args['device'])
      if len(prediction_batch.shape) == 3:
        prediction_batch = torch.argmax(prediction_batch, 2)
      else:
        prediction_batch = torch.argmax(prediction_batch, 1)
        label_batch = label_batch.view(label_batch.shape[0])
      agreements = (prediction_batch == label_batch).long()
      filtered_agreements = torch.where(label_batch != 0, agreements,
              torch.zeros_like(agreements))
      total_agreements = torch.sum(filtered_agreements.long())
      total_labels = torch.sum((label_batch != 0).long())
      total += total_labels.cpu().numpy()
      correct += total_agreements.cpu().numpy()

    with open(os.path.join(self.reporting_root, split_name + '.label_acc'), 'w') as fout:
      fout.write(str(float(correct)/  total) + '\n')

  def report_v_entropy(self, prediction_batches, dataset, split_name):
    total_label_count = 0
    neg_logprob_sum = 0
    for prediction_batch, (_, label_batch, sentences) in zip(prediction_batches, dataset):
      prediction_batch = prediction_batch.to(self.args['device'])
      batch_label_count = torch.sum((label_batch != 0).long())
      if len(prediction_batch.shape) == 3:
        prediction_batch = torch.softmax(prediction_batch, 2)
        label_batch = label_batch.view(*label_batch.shape, 1)
        prediction_batch = torch.gather(prediction_batch, 2, label_batch)
      else:
        prediction_batch  = torch.softmax(prediction_batch, 1)
        prediction_batch = torch.gather(prediction_batch, 1, label_batch)
        label_batch = label_batch.view(label_batch.shape[0])
        label_batch = label_batch.view(*label_batch.shape, 1)
      batch_neg_logprob_sum = -torch.sum(torch.where((label_batch!=0),
        torch.log2(prediction_batch), torch.zeros_like(prediction_batch)))

      total_label_count += batch_label_count
      neg_logprob_sum += batch_neg_logprob_sum

    with open(os.path.join(self.reporting_root, split_name + '.v_entropy'), 'w') as fout:
      fout.write(str(float(neg_logprob_sum)/float(total_label_count)) + '\n')

class NERReporter(IndependentLabelReporter):
  """ Class for reporting metrics on the Named Entity Recognition task.

  Requires special handling because of entity-level eval and integration
  with the stanza library scorer.
  """
  yaml_tag = '!NERReporter'

  def __init__(self, args, reporting_root, reporting_methods, ner_task):
    """
    Arguments:
      reporting_root: path to which results will be written
      reporting_methods: list of metrics to report
      ner_task: the NERClassificationTask object representing the NER task;
                used to map integer labels to label strings
    """
    self.args = args
    self.reporting_methods = reporting_methods
    self.reporting_method_dict = {
        'label_accuracy':self.report_label_values,
        'v_entropy':self.report_v_entropy,
        'ner_f1':self.report_ner_f1
        }
    #self.reporting_root = args['reporting']['root']
    self.reporting_root = reporting_root
    self.test_reporting_constraint = {'label_accuracy', 'v_entropy', 'ner_f1'}
    self.ner_task = ner_task


  def report_ner_f1(self, prediction_batches, dataset, split_name):
    """
    Reports entity-level NER F1 using the stanza library scorer
    """
    string_predictions = []
    string_labels = []
    for prediction_batch, (_, label_batch, sentences) in zip(prediction_batches, dataset):
      prediction_batch = prediction_batch.to(self.args['device'])
      prediction_batch = torch.argmax(prediction_batch, 2)
      for prediction_sentence, label_sentence in zip(prediction_batch, label_batch):
        string_predictions.append(list(filter(lambda x: x != '-', [self.ner_task.category_string_of_label_int(x)
          for x in prediction_sentence])))
        string_labels.append(list(filter(lambda x: x != '-', [self.ner_task.category_string_of_label_int(x)
          for x in label_sentence])))
    precision, recall, f1 = score_by_entity(string_predictions, string_labels)

    with open(os.path.join(self.reporting_root, split_name + '.f1'), 'w') as fout:
      fout.write(str(f1) + '\n')
    with open(os.path.join(self.reporting_root, split_name + '.precision'), 'w') as fout:
      fout.write(str(precision) + '\n')
    with open(os.path.join(self.reporting_root, split_name + '.recall'), 'w') as fout:
      fout.write(str(recall) + '\n')
