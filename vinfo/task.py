import os 

import torch
from stanza.models.ner.utils import bio2_to_bioes

from yaml import YAMLObject
from utils import IGNORE_LABEL_INDEX, TRAIN_STR, DEV_STR, TEST_STR, InitYAMLObject
from utils import get_conversion_dict
import h5py


class TokenClassificationTask(InitYAMLObject):
  """
  General form of a task that requires the categorization
  of each dataset-given token into one of a finite number of
  categories.
  """
  yaml_tag = '!TokenClassificationTask'
  train_cache = None
  dev_cache = None
  test_cache = None

  def __init__(self, args, task_name, input_fields, cache=None):
    """
    Args:
     - task_name: the string identifier for the task.
                  e.g., the field name in the ontonotes
                  annotation that provides labels for this task
    """
    self.task_name = task_name
    #self.name_to_index_dict = {name:i for i, name in enumerate(args['input_fields'])}
    self.label_vocab = {'[PAD]':0, '-':0, '_':0}
    self.ints_to_strings = {}
    #self.cache = cache
    self.cache = None
    self.input_fields = input_fields
    self.task_name = task_name
    self.name_to_index_dict = None


  def setup_cache(self):
    """Constructs reader of or writer to disk cache

    If cache is exists and is valid, constructs a reader of the cache,
    otherwise constructs a writer to cache features as they're constructed
    """
    train_cache_path = self.cache.get_cache_path_and_check(TRAIN_STR, self.task_name)
    dev_cache_path = self.cache.get_cache_path_and_check(DEV_STR, self.task_name)
    test_cache_path = self.cache.get_cache_path_and_check(TEST_STR, self.task_name)

    self.train_cache_writer = None
    self.dev_cache_writer = None
    self.test_cache_writer = None

    if os.path.exists(train_cache_path):
      f = h5py.File(train_cache_path, 'r')
      self.train_cache = (torch.tensor(f[str(i)][()]) for i in range(len(f.keys())))
    else:
      self.train_cache_writer = h5py.File(train_cache_path, 'w')
    if os.path.exists(dev_cache_path):
      f2 = h5py.File(dev_cache_path, 'r')
      self.dev_cache = (torch.tensor(f2[str(i)][()]) for i in range(len(f2.keys())))
    else:
      self.dev_cache_writer = h5py.File(dev_cache_path, 'w')
    if os.path.exists(test_cache_path):
      f3 = h5py.File(test_cache_path, 'r')
      self.test_cache = (torch.tensor(f3[str(i)][()]) for i in range(len(f3.keys())))
    else:
      self.test_cache_writer = h5py.File(test_cache_path, 'w')

  def category_int_of_label_string(self, label_string):
    """ Constructs and accesses label vocab

    Arguments:
      label_string: the string representation of an annotation label
    Output:
      The index of that label if it exists, or the newly assigned index 
      for that label of one did not previously exist
    """
    if label_string not in self.label_vocab:
      self.label_vocab[label_string] = max(self.label_vocab.values())+1
    return self.label_vocab[label_string]

  def category_string_of_label_int(self, label_integer):
    """ Convergs from label integers back to their strings

    Arguments:
      label_string: the integer representation of an annotation label
    Output:
      The string of that label if it exists, or raises an error
    """
    if len(self.ints_to_strings) < len(self.label_vocab):
      self.ints_to_strings = {index: label for (label, index) in self.label_vocab.items()}
    return self.ints_to_strings[int(label_integer)]


  def _manual_setup(self):
    """ Handles initialization not done in __init__ because of the YAML constructor

    Done when labels_of_sentence is called, but if (for testing) another function is
    called, there may be extra setup to be done. (Not done in init because the yaml
    constructor doesn't guarantee that all arguments will be present)
    """
    # If self.cache is None, then all caching should be skipped
    if self.name_to_index_dict is None:
      self.name_to_index_dict = {name:i for i, name in enumerate(self.input_fields)}
      self.task_label_index = self.name_to_index_dict[self.task_name]
      if self.cache is not None:
        self.setup_cache()

  def labels_of_sentence(self, sentence, split):
    """ Provides a tensor of labels for a sentence

    Arguments:
      sentence: a list of lists of data read with fields given in args['input_fields']
      split: the split ('train','dev','test') that this data belongs to, for caching.
    Output:
      a tensor with a label for each token in the sentence as given by the annotation
      for the task specified by self.task_name
    """
    self._manual_setup()

    if self.cache is None:
      labels = self._labels_of_sentence(sentence, split)
      return labels

    # Otherwise, either read from or write to cache
    if split == TRAIN_STR and self.train_cache:
      return next(self.train_cache)
    if split == DEV_STR and self.dev_cache:
      return next(self.dev_cache)
    if split == TEST_STR and self.test_cache:
      return next(self.test_cache)
    cache_writer = (self.train_cache_writer if split == TRAIN_STR else (
                    self.dev_cache_writer if split == DEV_STR else (
                    self.test_cache_writer if split == TEST_STR else None)))
    if cache_writer is None:
      raise ValueError("Unknown split: {}".format(split))
    labels = self._labels_of_sentence(sentence, split)
    string_key = str(len(cache_writer.keys()))
    dset = cache_writer.create_dataset(string_key, labels.shape)
    dset[:] = labels
    return labels

  def _labels_of_sentence(self, sentence, esplit):
    """ Provides a tensor of labels for a sentence; no caching

    Arguments:
      sentence: a list of lists of data read with fields given in args['input_fields']
    Output:
      a tensor with a label for each token in the sentence as given by the annotation
      for the task specified by self.task_name
    """

    labels = torch.zeros(len(sentence))
    for token_index, token_attribute_list in enumerate(sentence):
      label_string = token_attribute_list[self.task_label_index]
      labels[token_index] = self.category_int_of_label_string(label_string)
    return labels

class CoarseTokenClassificationTask(TokenClassificationTask):
  yaml_tag = '!CoarseTokenClassificationTask'
  train_cache = None
  dev_cache = None
  test_cache = None

  def __init__(self, args, task_name, input_fields, conversion_name, cache=None, train_types_only=False):
    """
    Task which annotates coarse versions of existing annotations
    from a dataset, using one of a hard-coded set of conversion
    dictionaries.

    Args:
     - input_fields: the list of column labels for the input dataset.
     - task_name: used as the fine-grained tags for conversion
    """
    self.input_fields = input_fields
    self.task_name = task_name
    self.conversion_name = conversion_name
    self.conversion_dict = None
    self.name_to_index_dict = None
    self.cache = None
    self.ints_to_strings = {}
    self.label_vocab = {'[PAD]':0, '-':0, '_':0}
    self.train_types_only = train_types_only
    self.train_type_vocab = set()

  def category_int_of_label_string(self, label_string):
    """ Constructs and accesses label vocab, with coarsening

    Provides label vocab but maps each given label using the
    dictionary provided by the conversion name

    Arguments:
      label_string: the string representation of an annotation label
    Output:
      The index of that label if it exists, or the newly assigned index 
      for that label of one did not previously exist
    """
    if self.conversion_dict is None:
      self.conversion_dict = get_conversion_dict(self.conversion_name)
    label_string = self.conversion_dict[label_string]
    if label_string not in self.label_vocab:
      self.label_vocab[label_string] = len(self.label_vocab)
    return self.label_vocab[label_string]

  def _labels_of_sentence(self, sentence, split):
    """ Provides a tensor of labels for a sentence; no caching

    Optionally limits

    Arguments:
      sentence: a list of lists of data read with fields given in args['input_fields']
    Output:
      a tensor with a label for each token in the sentence as given by the annotation
      for the task specified by self.task_name
    """

    labels = torch.zeros(len(sentence))
    for token_index, token_attribute_list in enumerate(sentence):
      label_string = token_attribute_list[self.task_label_index]
      if self.train_types_only:
        word_type = token_attribute_list[self.input_fields.index('token')]
        if split == TRAIN_STR:
          self.train_type_vocab.add(word_type)
        else:
          old_label_string = label_string
          label_string = label_string if word_type in self.train_type_vocab else '-'
      labels[token_index] = self.category_int_of_label_string(label_string)
    return labels

class NERClassificationTask(TokenClassificationTask):

  yaml_tag = '!NERClassificationTask'
  train_cache = None
  dev_cache = None
  test_cache = None

  def __init__(self, args, task_name, input_fields, cache=None):
    """
    Args:
     - task_name: the string identifier for the task.
                  e.g., the field name in the ontonotes
                  annotation that provides labels for this task
    """
    self.task_name = task_name
    #self.name_to_index_dict = {name:i for i, name in enumerate(args['input_fields'])}
    self.label_vocab = {'[PAD]':0, '-':0, 'O':1, '_':0}
    #self.cache = cache
    self.cache = None
    self.input_fields = input_fields
    self.task_name = task_name
    self.name_to_index_dict = None
    self.ints_to_strings = {}

  def _string_labels_of_sentence(self, sentence):
    """ Provides the BIOES string annotation for NER from the annotations of Ontonotes
    """
    label_strings = []
    ongoing_label = 'O'
    for token_index, token_attribute_list in enumerate(sentence):
      raw_label_string = token_attribute_list[self.task_label_index].strip('*')
      if '(' in raw_label_string:
        ongoing_label = raw_label_string.strip('(').strip(')')
        beginning = True
      #labels[token_index] = self.category_int_of_label_string(ongoing_label)
      if ongoing_label == 'O':
        label_strings.append(ongoing_label)
      else:
        label_strings.append('{}-{}'.format('B' if beginning else 'I', ongoing_label))
      beginning = False
      if ')' in raw_label_string:
        ongoing_label = 'O'
    #bioes_tags = bio2_to_bioes(label_strings)
    bioes_tags = label_strings
    return bioes_tags

  def _labels_of_sentence(self, sentence, split):
    """ Provides a tensor of labels for a sentence; no caching

    Arguments:
      sentence: a list of lists of data read with fields given in args['input_fields']
    Output:
      a tensor with a label for each token in the sentence as given by the annotation
      for the task specified by self.task_name
    """
    #print(self.label_vocab)
    self.category_int_of_label_string('O')
    bioes_tags = self._string_labels_of_sentence(sentence)
    labels = torch.zeros(len(sentence))
    for index, label in enumerate(bioes_tags):
      labels[index] = self.category_int_of_label_string(label)
    return labels

class SentenceClassificationTask(TokenClassificationTask):

  yaml_tag = '!SentenceClassificationTask'
  train_cache = None
  dev_cache = None
  test_cache = None

  def __init__(self, args, task_name, cache=None):
    """
    
    """
    #self.task_name = task_name
    #self.name_to_index_dict = {name:i for i, name in enumerate(args['input_fields'])}
    self.name_to_index_dict = {'index': 0, 'token': 1, 'label': 2}
    self.task_name = task_name
    self.input_fields = ['index', 'token', 'label']
    self.label_vocab = {'[PAD]':0}
    #self.cache = cache
    self.cache = None
    self.ints_to_strings = {}
    #self.input_fields = input_fields
    #self.task_name = task_name
    #self.name_to_index_dict = None

  def _labels_of_sentence(self, sentence, split):
    """ Provides a tensor of labels for a sentence; no caching

    Arguments:
      sentence: a list of lists of data where the fields are (token, sentence_label)
    Output:
      a tensor with a single label for the sentence (shape 1)
    """
    labels = torch.ones(1)
    labels[0] = self.category_int_of_label_string(sentence[0][self.name_to_index_dict['label']]) #
    return labels

