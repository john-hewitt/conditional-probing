import os
import h5py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset, DataLoader
import Levenshtein as levenshtein

from tqdm import tqdm
from yaml import YAMLObject
from transformers import AutoTokenizer, AutoModel
from allennlp.modules.elmo import batch_to_ids

from utils import TRAIN_STR, DEV_STR, TEST_STR, InitYAMLObject

BATCH_SIZE = 50
"""
Classes for loading, caching, and yielding text datasets
"""


#class Dataset(Dataset, InitYAMLObject):
#  """
#  Base class for objects that serve batches of
#  tensors. For decoration/explanation only
#  """
#  yaml_tag = '!Dataset'

class IterableDatasetWrapper(Dataset):#(IterableDataset):
  """
  Wrapper class to pass to a DataLoader so it doesn't
  think the underlying generator should have a len() fn.

  But I gave up on this for various reasons so it's just
  a normal dataset, here in case I try again.
  """
  def __init__(self, generator):
    self.generator = generator #[x for x in generator]
  def __iter__(self):
    return iter(self.generator)
  def __len__(self):
    return len(self.generator)
  def __getitem__(self, idx):
    return self.generator[idx]

class ListDataset(Dataset, InitYAMLObject):
  """
  Container class for collecting multiple annotation or 
  representation datasets and a single target task dataset
  , and serving all of them
  """
  yaml_tag = '!ListDataset'
  def __init__(self, args, data_loader, output_dataset, input_datasets):
    """
    Arguments:
      output_datset: 
    """
    self.args = args
    self.input_datasets = input_datasets
    self.output_dataset = output_dataset
    self.data_loader = data_loader
    self.train_data = None
    self.dev_data = None
    self.test_data = None

  def get_train_dataloader(self, shuffle=True):
    """Returns a PyTorch DataLoader object with the training data
    """
    if self.train_data is None:
      self.train_data = list(self.load_data(TRAIN_STR))
      #generator = IterableDatasetWrapper(self.load_data(TRAIN_STR))
    generator = IterableDatasetWrapper(self.train_data)
    return DataLoader(generator, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=self.collate_fn)

  def get_dev_dataloader(self, shuffle=False):
    """Returns a PyTorch DataLoader object with the dev data
    """
    if self.dev_data is None:
      self.dev_data = list(self.load_data(DEV_STR))
      #generator = IterableDatasetWrapper(self.load_data(DEV_STR))
    generator = IterableDatasetWrapper(self.dev_data)
    return DataLoader(generator, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=self.collate_fn)

  def get_test_dataloader(self, shuffle=False):
    """Returns a PyTorch DataLoader object with the test data
    """
    if self.test_data is None:
      self.test_data = list(self.load_data(TEST_STR))
      #generator = IterableDatasetWrapper(self.load_data(TEST_STR))
    generator = IterableDatasetWrapper(self.test_data)
    return DataLoader(generator, batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=self.collate_fn)

  def load_data(self, split_string):
    """Loads data from disk into RAM tensors for passing to a network on GPU

    Iterates through the training set once, passing each sentence to each
    input Dataset and the output Dataset
    """
    for sentence in tqdm(self.data_loader.yield_dataset(split_string),desc='[loading]'):
      input_tensors = []
      for dataset in self.input_datasets:
        input_tensors.append(dataset.tensor_of_sentence(sentence, split_string))
      output_tensor = self.output_dataset.tensor_of_sentence(sentence, split_string)
      yield (input_tensors, output_tensor, sentence)

  def collate_fn(self, observation_list):
    """
    Combines observations (input_tensors, output_tensor, sentence) tuples
    input_tensors is of the form ((annotation, alignment), ..., (annotation, alignment))
    output_tensor is of the form (annotation, alignment),

    to batches of observations ((batches_input_1, batches_input_2), batches_output, sentences)
    """
    sentences = (x[2] for x in observation_list)
    max_corpus_token_len = max((len(x) for x in sentences))
    input_annotation_tensors = []
    input_alignment_tensors = []
    input_tensor_count = len(observation_list[0][0])
    for input_tensor_index in range(input_tensor_count):
      max_annotation_token_len = max([x[0][input_tensor_index][0].shape[0] for x in observation_list])
      intermediate_annotation_list = []
      intermediate_alignment_list = []
      for input_annotation, input_alignment in ((x[0][input_tensor_index][0], 
          x[0][input_tensor_index][1]) for x in observation_list):
        if len(input_annotation.shape) == 1: # word-level ids
          new_annotation_tensor = torch.zeros(max_annotation_token_len, dtype=torch.long)
          new_annotation_tensor[:len(input_annotation)] = input_annotation
        elif len(input_annotation.shape) == 2: # characeter-level ids
          new_annotation_tensor = torch.zeros(max_annotation_token_len, input_annotation.shape[1]).long()
          new_annotation_tensor[:len(input_annotation),:] = input_annotation
        intermediate_annotation_list.append(new_annotation_tensor)
        new_alignment_tensor = torch.zeros(max_annotation_token_len, max_corpus_token_len)
        new_alignment_tensor[:input_alignment.shape[0], :input_alignment.shape[1]] = input_alignment
        intermediate_alignment_list.append(new_alignment_tensor)
      input_annotation_tensors.append(torch.stack(intermediate_annotation_list).to(self.args['device']))
      input_alignment_tensors.append(torch.stack(intermediate_alignment_list).to(self.args['device']))

    intermediate_annotation_list = []
    intermediate_alignment_list = []
    max_output_annotation_len = max([x[1][0].shape[0] for x in observation_list])
    for output_annotation, output_alignment in (x[1] for x in observation_list):
      new_annotation_tensor = torch.zeros(max_output_annotation_len, dtype=torch.long)
      new_annotation_tensor[:len(output_annotation)] = output_annotation
      intermediate_annotation_list.append(new_annotation_tensor)
    output_annotation_tensor = torch.stack(intermediate_annotation_list).to(self.args['device'])
    sentences = [x[2] for x in observation_list]
    return ((input_annotation_tensors, input_alignment_tensors), output_annotation_tensor, sentences)

class ELMoData(InitYAMLObject):
    """
    Loading and serving minibatches of tokens to input to
    ELMo, as mediated by allennlp.
    """
    yaml_tag = '!ELMoData'
    def __init__(self, args):
      self.args = args

    def tensor_of_sentence(self, sentence, split_string):
      """
      Provides character indices for a single sentence.
      """
      words = [x[1] for x in sentence]
      alignment = torch.eye(len(words))
      return batch_to_ids([words])[0,:,:], alignment
      #for index, token in enumerate([x[1] for x in sentence]):


    

class HuggingfaceData(InitYAMLObject):
  """
  Loading and serving minibatches of tokens to input
  to a Huggingface-loaded model.
  """
  yaml_tag = '!HuggingfaceData'
  def __init__(self, args, model_string, cache=None):
    print('Constructing HuggingfaceData of {}'.format(model_string))
    self.tokenizer = AutoTokenizer.from_pretrained(model_string) #, add_prefix_space=True)
    self.args = args
    self.cache = cache
    self.task_name = 'hfacetokens.{}'.format(model_string)
    self.cache_is_setup = False

  def levenshtein_matrix(self, string1, string2):
    opcodes = levenshtein.opcodes(string1, string2)
    mtx = torch.zeros(len(string1), len(string2))
    cumulative = 0
    for opcode in opcodes:
      opcode_type, str1b, str1e, str2b, str2e = opcode
      if opcode_type in {'equal', 'replace'}:
        diff = str1e - str1b
        for i in range(diff):
          mtx[str1b+i,str2b+i] = 1
      if opcode_type == 'delete':
        diff = str1e - str1b
        for i in range(diff):
          mtx[str1b+i, str2b] = 1
      if opcode_type == 'insert':
        diff = str2e - str2b
        for i in range(diff):
          mtx[str1b, str2b+i] = 1
    return mtx
  
  def token_to_character_alignment(self, tokens):
    ptb_sentence_length = sum((len(tok) for tok in tokens))
    ptb_string_token_alignment = []
    cumulative = 0
    for token in tokens:
      new_alignment = torch.zeros(ptb_sentence_length)
      for i, char in enumerate(token):
        if char == ' ':
          continue
        new_alignment[i+cumulative] = 1
      new_alignment = new_alignment / sum(new_alignment)
      cumulative += len(token)
      ptb_string_token_alignment.append(new_alignment)
    return torch.stack(ptb_string_token_alignment)
  
  def de_ptb_tokenize(self, tokens):
    tokens_with_spaces = []
    new_tokens_with_spaces = []
    ptb_sentence_length = sum((len(tok) for tok in tokens))
    token_alignments = []
  
    cumulative = 0
    for i, _ in enumerate(tokens):
      token = tokens[i]
      next_token = tokens[i+1] if i < len(tokens)-1 else '<EOS>'
      # Handle LaTeX-style quotes
      if token.strip() in {"``", "''"}:
        new_token = '"'
      elif token.strip() == '-LRB-':
        new_token = '('
      elif token.strip() == '-RRB-':
        new_token = ')'
      elif token.strip() == '-LSB-':
        new_token = '['
      elif token.strip() == '-RSB-':
        new_token = ']'
      elif token.strip() == '-LCB-':
        new_token = '{'
      elif token.strip() == '-RCB-':
        new_token = '}'
      else:
        new_token = token
      use_space = (token.strip() not in {'(', '[', '{', '"', "'", '``', "''"} and
                  next_token.strip() not in {"'ll", "'re", "'ve", "n't",
                                "'s", "'LL", "'RE", "'VE",
                                "N'T", "'S", '"', "'", '``', "''", ')', '}', ']',
                                '.', ';', ':', '!', '?'}
                  and i != len(tokens) - 1)
  
      new_token = new_token.strip() + (' ' if use_space else '')
      new_tokens_with_spaces.append(new_token)
      tokens_with_spaces.append(token)
  
      new_alignment = torch.zeros(ptb_sentence_length)
      for index, char in enumerate(token):
        new_alignment[index+cumulative] = 1
      #new_alignment = new_alignment / sum(new_alignment)
      for new_char in new_token:
        token_alignments.append(new_alignment)
      cumulative += len(token)
    return new_tokens_with_spaces, torch.stack(token_alignments)
  
  def hface_ontonotes_alignment(self, sentence):
    tokens = [x[1] for x in sentence]
    tokens = [ x + (' ' if i !=len(tokens)-1 else '') for (i, x) in enumerate(tokens)]
    raw_tokens, ptb_to_deptb_alignment = self.de_ptb_tokenize(tokens)
    raw_string = ''.join(raw_tokens)
    ptb_token_to_ptb_string_alignment = self.token_to_character_alignment(tokens)
    #tokenizer = transformers.AutoTokenizer.from_pretrained('roberta-base')
    hface_tokens = self.tokenizer.tokenize(raw_string)
    hface_tokens_with_spaces = [x+ (' ' if i != len(hface_tokens)-1 else '')for (i, x) in enumerate(hface_tokens)]
    hface_token_to_hface_string_alignment = self.token_to_character_alignment(hface_tokens_with_spaces)
    hface_string = ' '.join(hface_tokens)
    hface_character_to_deptb_character_alignment = self.levenshtein_matrix(hface_string, raw_string)
    unnormalized_alignment = torch.matmul(torch.matmul(hface_token_to_hface_string_alignment.to(self.args['device']), hface_character_to_deptb_character_alignment.to(self.args['device'])),
          torch.matmul(ptb_token_to_ptb_string_alignment.to(self.args['device']), ptb_to_deptb_alignment.to(self.args['device']).t()).t())
    return (unnormalized_alignment / torch.sum(unnormalized_alignment, dim=0)).cpu(), hface_tokens, raw_string

  def _setup_cache(self):
    """
    Constructs readers for caches that exist
    and writers for caches that do not.
    """
    if self.cache is None:
      return
    if self.cache_is_setup:
      return

    # Check cache readable/writeable
    train_cache_path, train_cache_readable, train_cache_writeable = \
        self.cache.get_cache_path_and_check(TRAIN_STR, self.task_name)
    dev_cache_path, dev_cache_readable, dev_cache_writeable = \
        self.cache.get_cache_path_and_check(DEV_STR, self.task_name)
    test_cache_path, test_cache_readable, test_cache_writeable = \
        self.cache.get_cache_path_and_check(TEST_STR, self.task_name)

    # If any of the train/dev/test are neither readable nor writeable, do not use cache.
    if ((not train_cache_readable and not train_cache_writeable) or
        (not dev_cache_readable and not dev_cache_writeable) or
        (not test_cache_readable and not test_cache_writeable)): 
      self.cache = None
      print("Not using the cache at all, since at least of one "
            "of {train,dev,test} cache neither readable nor writable.")
      return

    # Load readers or writers
    self.train_cache_writer = None
    self.dev_cache_writer = None
    self.test_cache_writer = None

    if train_cache_readable:
      f = h5py.File(train_cache_path, 'r')
      self.train_cache_tokens = (torch.tensor(f[str(i)+'tok'][()]) for i in range(len(f.keys())))
      self.train_cache_alignments = (torch.tensor(f[str(i)+'aln'][()]) for i in range(len(f.keys())))
    elif train_cache_writeable:
      #self.train_cache_writer = h5py.File(train_cache_path, 'w')
      self.train_cache_writer = self.cache.get_hdf5_cache_writer(train_cache_path)
      self.train_cache_tokens = None
      self.train_cache_alignments = None
    else:
      raise ValueError("Train cache neither readable nor writeable")
    if dev_cache_readable:
      f2 = h5py.File(dev_cache_path, 'r')
      self.dev_cache_tokens = (torch.tensor(f2[str(i)+'tok'][()]) for i in range(len(f2.keys())))
      self.dev_cache_alignments = (torch.tensor(f2[str(i)+'aln'][()]) for i in range(len(f2.keys())))
    elif dev_cache_writeable:
      #self.dev_cache_writer = h5py.File(dev_cache_path, 'w')
      self.dev_cache_writer = self.cache.get_hdf5_cache_writer(dev_cache_path)
      self.dev_cache_tokens = None
      self.dev_cache_alignments = None
    else:
      raise ValueError("Dev cache neither readable nor writeable")
    if test_cache_readable:
      f3 = h5py.File(test_cache_path, 'r')
      self.test_cache_tokens = (torch.tensor(f3[str(i)+'tok'][()]) for i in range(len(f3.keys())))
      self.test_cache_alignments = (torch.tensor(f3[str(i)+'aln'][()]) for i in range(len(f3.keys())))
    elif test_cache_writeable:
      #self.test_cache_writer = h5py.File(test_cache_path, 'w')
      self.test_cache_writer = self.cache.get_hdf5_cache_writer(test_cache_path)
      self.test_cache_tokens = None
      self.test_cache_alignments = None
    else:
      raise ValueError("Test cache neither readable nor writeable")
    self.cache_is_setup = True
    

  def tensor_of_sentence(self, sentence, split):
    self._setup_cache()
    if self.cache is None:
      labels = self._tensor_of_sentence(sentence, split)
      return labels

    # Otherwise, either read from or write to cache
    if split == TRAIN_STR and self.train_cache_tokens is not None:
      return next(self.train_cache_tokens), next(self.train_cache_alignments)
    if split == DEV_STR and self.dev_cache_tokens is not None:
      return next(self.dev_cache_tokens), next(self.dev_cache_alignments)
    if split == TEST_STR and self.test_cache_tokens is not None:
      return next(self.test_cache_tokens), next(self.test_cache_alignments)
    cache_writer = (self.train_cache_writer if split == TRAIN_STR else (
                    self.dev_cache_writer if split == DEV_STR else (
                    self.test_cache_writer if split == TEST_STR else None)))
    if cache_writer is None:
      raise ValueError("Unknown split: {}".format(split))
    wordpiece_indices, alignments = self._tensor_of_sentence(sentence, split)

    tok_string_key = str(len(list(filter(lambda x: 'tok' in x, cache_writer.keys())))) + 'tok'
    tok_dset = cache_writer.create_dataset(tok_string_key, wordpiece_indices.shape)
    tok_dset[:] = wordpiece_indices

    aln_string_key = str(len(list(filter(lambda x: 'aln' in x, cache_writer.keys())))) + 'aln'
    aln_dset = cache_writer.create_dataset(aln_string_key, alignments.shape)
    aln_dset[:] = alignments

    return wordpiece_indices, alignments


  def _tensor_of_sentence(self, sentence, split):
    alignment, wordpiece_strings, raw_string = self.hface_ontonotes_alignment(sentence)
    # add [SEP] and [CLS] empty alignments
    empty = torch.zeros(1, alignment.shape[1])
    alignment = torch.cat((empty, alignment, empty))
    #wordpiece_indices = torch.tensor(self.tokenizer(wordpiece_strings)
    wordpiece_indices = torch.tensor(self.tokenizer(raw_string).input_ids) #, is_split_into_words=True))
    return wordpiece_indices, alignment

  def _naive_tensor_of_sentence(self, sentence, split_string):
    """
    Converts from a tuple-formatted sentence (e.g, from conll-formatted data)
    to a Torch tensor of integers representing subword piece ids for input to
    a Huggingface-formatted neural model
    """
    # CLS token given by tokenizer
    wordpiece_indices = []
    wordpiece_alignment_vecs = [torch.zeros(len(sentence))]
    # language tokens
    for index, token in enumerate([x[1] for x in sentence]):
      new_wordpieces = self.tokenizer.tokenize(token)
      wordpiece_alignment = torch.zeros(len(sentence))
      wordpiece_alignment[index] = 1
      for wordpiece in new_wordpieces:
        wordpiece_alignment_vecs.append(torch.clone(wordpiece_alignment))
      wordpiece_indices.extend(new_wordpieces)
    # SEP token given by tokenizer
    wordpiece_indices = torch.tensor(self.tokenizer.encode(wordpiece_indices))
    wordpiece_alignment_vecs.append(torch.zeros(len(sentence)))
    wordpiece_alignment_vecs = torch.stack(wordpiece_alignment_vecs)
    return wordpiece_indices, wordpiece_alignment_vecs

class AnnotationData(InitYAMLObject):
  """
  Loading and serving minibatches of data from annotations
  """
  yaml_tag = '!AnnotationDataset'
  def __init__(self, args, task):
    self.args = args
    self.task = task
    #self.task.setup_cache()

  def tensor_of_sentence(self, sentence, split_string):
    """
    Converts from a tuple-formatted sentence (e.g, from conll-formatted data)
    to a Torch tensor of integers representing the annotation
    """
    alignment = torch.eye(len(sentence))
    return self.task.labels_of_sentence(sentence, split_string), alignment


class Loader(InitYAMLObject):
  """
  Base class for objects that read datasets from disk
  and yield sentence buffers for tokenization and labeling
  Strictly for description
  """
  yaml_tag = '!Loader'


class OntonotesReader(Loader):
  """
  Minutae for reading the Ontonotes dataset,
  as formatted as described in the readme
  """
  yaml_tag = '!OntonotesReader'

  def __init__(self, args, train_path, dev_path, test_path, cache):
    print('Constructing OntoNotesReader')
    self.train_path = train_path
    self.dev_path = dev_path
    self.test_path = test_path
    self.cache = cache


  @staticmethod
  def sentence_lists_of_stream(ontonotes_stream):
    """
    Yield sentences from raw ontonotes stream

    Arguments:
      ontonotes_stream: iterable of ontonotes file lines
    Yields:
      a buffer for each sentence in the stream; elements
      in the buffer are lists defined by TSV fields of the
      ontonotes stream
    """
    buf = []
    for line in ontonotes_stream:
      if line.startswith('#'):
        continue
      if not line.strip():
        yield buf
        buf = []
      else:
        buf.append([x.strip() for x in line.split('\t')])
    if buf:
      yield buf

  def yield_dataset(self, split_string):
    """
    Yield a list of attribute lines, given by ontonotes_fields,
    for each sentence in the training set of ontonotes
    """
    path = (self.train_path if split_string == TRAIN_STR else
            (self.dev_path if split_string == DEV_STR else
            (self.test_path if split_string == TEST_STR else
            None)))
    if path is None:
      raise ValueError("Unknown split string: {}".format(split_string))

    with open(path) as fin:
      for sentence in OntonotesReader.sentence_lists_of_stream(fin):
        yield sentence


class SST2Reader(Loader):
  """
  Minutae for reading the Stanford Sentiment (SST-2)
  dataset, as downloaded from the GLUE website.
  """
  yaml_tag = '!SST2Reader'

  def __init__(self, args, train_path, dev_path, test_path, cache):
    print('Constructing SST2Reader')
    self.train_path = train_path
    self.dev_path = dev_path
    self.test_path = test_path
    self.cache = cache

  @staticmethod
  def sentence_lists_of_stream(sst2_stream):
    """
    Yield sentences from raw sst2 stream

    Arguments:
      sst2_stream: iterable of sst2_stream lines
    Yields:
      a buffer for each sentence in the stream;
      elements in the buffer are lists defined by TSV
      fields of the ontonotes stream
    """
    _ = next(sst2_stream) # Get rid of the column labels 
    for line in sst2_stream:
      word_string, label_string = [x.strip() for x in line.split('\t')]
      word_tokens = word_string.split(' ')
      indices = [str(i) for i, _ in enumerate(word_tokens)]
      label_tokens = [label_string for _ in word_tokens]
      yield list(zip(indices, word_tokens, label_tokens))

  def yield_dataset(self, split_string):
    """
    Yield a list of attribute lines, given by ontonotes_fields,
    for each sentence in the training set of ontonotes
    """
    path = (self.train_path if split_string == TRAIN_STR else
            (self.dev_path if split_string == DEV_STR else
            (self.test_path if split_string == TEST_STR else
            None)))
    if path is None:
      raise ValueError("Unknown split string: {}".format(split_string))

    with open(path) as fin:
      for sentence in SST2Reader.sentence_lists_of_stream(fin):
        yield sentence
