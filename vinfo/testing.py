import unittest
import tempfile
import glob
import os

import torch
import torch.nn as nn
import numpy as np

import model
import dataset
import task
import cache
import probe
import trainer
import reporter
from utils import TRAIN_STR, DEV_STR, TEST_STR

"""
Test suites for the codebase
"""

class DataTest(unittest.TestCase):

  def assert_2darray_equal(self, gold_sentence, other_sentence):
    """
    Sentences are effectively 2D arrays of strings;
    this checks whether they contain the same objects
    """
    for index, array in enumerate(gold_sentence):
      for index2, elt in enumerate (array):
        self.assertEqual(elt, other_sentence[index][index2])

  def assert_3darray_equal(self, gold_sentence, other_sentence):
    """
    Sentences are effectively 2D arrays of strings;
    this checks whether they contain the same objects
    """
    for index, array in enumerate(gold_sentence):
      for index2, array2 in enumerate (array):
        for index3, elt in enumerate (array2):
          self.assertEqual(elt, other_sentence[index][index2][index3])

  def assert_annotations_equal(self, gold_annotation, other_annotation):
    """
    Annotations are effectively arrays;
    this checks whether they contain the same values
    """
    for index, elt in enumerate(gold_annotation):
        self.assertEqual(elt, other_annotation[index])

class OntonotesReaderTest(DataTest):

  def setUp(self):
    """
    Writes temporary files with simple Ontonotes-formatted data
    """
    self.tmpfile1 = tempfile.NamedTemporaryFile(mode='w')
    self.tmpfile1.write("""I\tNoun\nenjoy\tVerb\npizza\tNoun""")
    self.tmpfile1.flush()
    self.tmpfile2 = tempfile.NamedTemporaryFile(mode='w')
    self.tmpfile2.write("""I\tNoun\nenjoy\tVerb\npizza\tNoun\n\nIt\tDet\nis\tVerb\ntasty\tAdj\n\n""")
    self.tmpfile2.flush()

  def tearDown(self):
    self.tmpfile1.close()
    self.tmpfile2.close()


  def test_one_sentence(self):
    """
    Tests that a one-sentence dataset can be read directly;
    the dataset does not have a blank line at the end;
    """
    gold_sentence = (('I', 'Noun'), ('enjoy', 'Verb'), ('pizza', 'Noun'))
    generator = dataset.OntonotesReader(None, self.tmpfile1.name, None, None, None).yield_dataset(TRAIN_STR)
    self.assert_2darray_equal(gold_sentence, next(generator))
    with self.assertRaises(StopIteration):
      next(generator)
    generator = dataset.OntonotesReader(None, None, self.tmpfile1.name, None, None).yield_dataset(DEV_STR)
    self.assert_2darray_equal(gold_sentence, next(generator))
    with self.assertRaises(StopIteration):
      next(generator)
    generator = dataset.OntonotesReader(None, None, None, self.tmpfile1.name, None).yield_dataset(TEST_STR)
    self.assert_2darray_equal(gold_sentence, next(generator))
    with self.assertRaises(StopIteration):
      next(generator)

  def test_two_sentences_with_gap(self):
    """
    Tests that a tw-sentence dataset can be read directly;
    the dataset has an extra blank line at the end that must
    be discarded.
    """
    gold_sentence1 = (('I', 'Noun'), ('enjoy', 'Verb'), ('pizza', 'Noun'))
    gold_sentence2 = (('It', 'Det'), ('is', 'Verb'), ('tasty', 'Adj'))
    generator = dataset.OntonotesReader(None, self.tmpfile2.name, None, None, None).yield_dataset(TRAIN_STR)
    self.assert_2darray_equal(gold_sentence1, next(generator))
    self.assert_2darray_equal(gold_sentence2, next(generator))
    with self.assertRaises(StopIteration):
      next(generator)
    generator = dataset.OntonotesReader(None, None, self.tmpfile2.name, None, None).yield_dataset(DEV_STR)
    self.assert_2darray_equal(gold_sentence1, next(generator))
    self.assert_2darray_equal(gold_sentence2, next(generator))
    with self.assertRaises(StopIteration):
      next(generator)
    generator = dataset.OntonotesReader(None, None, None, self.tmpfile2.name, None).yield_dataset(TEST_STR)
    self.assert_2darray_equal(gold_sentence1, next(generator))
    self.assert_2darray_equal(gold_sentence2, next(generator))
    with self.assertRaises(StopIteration):
      next(generator)

class TokenClassificationTaskTest(DataTest):

  def setUp(self):
    """
    Writes temporary files with simple Ontonotes-formatted data
    """
    self.tmpfile1 = tempfile.NamedTemporaryFile(mode='w')
    self.tmpfile1.write("""I\tNoun\nenjoy\tVerb\npizza\tNoun""")
    self.tmpfile1.flush()
    self.tmpfile2 = tempfile.NamedTemporaryFile(mode='w')
    self.tmpfile2.write("""I\tNoun\nenjoy\tVerb\npizza\tNoun\n\nIt\tDet\nis\tVerb\ntasty\tAdj\n\n""")
    self.tmpfile2.flush()
    self.tmpfile3 = tempfile.NamedTemporaryFile(mode='w')
    self.tmpfile3.write("""Good\tAdj\n""")
    self.tmpfile3.flush()

  def tearDown(self):
    for cache_filename in glob.glob(self.tmpfile1.name + '.cache.*'):
      os.remove(cache_filename)
    self.tmpfile1.close()
    for cache_filename in glob.glob(self.tmpfile2.name + '.cache.*'):
      os.remove(cache_filename)
    self.tmpfile2.close()
    for cache_filename in glob.glob(self.tmpfile3.name + '.cache.*'):
      os.remove(cache_filename)
    self.tmpfile3.close()

  def assert_annotations_equal(self, gold_annotation, other_annotation):
    """
    Annotations are effectively arrays;
    this checks whether they contain the same values
    """
    for index, elt in enumerate(gold_annotation):
        self.assertEqual(elt, other_annotation[index])

  def test_no_cache_labels_of_sentence(self):
    """
    Tests whether attributes can be accessed by name by the 
    TokenClassificationTask annotator
    """
    sentence1 = (('I', 'Noun'), ('enjoy', 'Verb'), ('pizza', 'Noun'))
    token_annotation_sentence_1 = [1, 2, 3]
    pos_annotation_sentence_1 = [1, 2, 1]

    sentence2 = (('It', 'Det'), ('is', 'Verb'), ('tasty', 'Adj'))
    token_annotation_sentence_2 = [4, 5, 6]
    pos_annotation_sentence_2 = [3, 2, 4]

    args = {'device':'cpu'}
    input_fields = ['Token', 'PoS']

    token_task = task.TokenClassificationTask(args, 'Token', input_fields)
    self.assert_annotations_equal(token_annotation_sentence_1, token_task.labels_of_sentence(sentence1, TRAIN_STR))
    self.assert_annotations_equal(token_annotation_sentence_2, token_task.labels_of_sentence(sentence2, TRAIN_STR))

    pos_task = task.TokenClassificationTask(args, 'PoS', input_fields)
    self.assert_annotations_equal(pos_annotation_sentence_1, pos_task.labels_of_sentence(sentence1, TRAIN_STR))
    self.assert_annotations_equal(pos_annotation_sentence_2, pos_task.labels_of_sentence(sentence2, TRAIN_STR))

  def test_make_cache_labels_of_sentence(self):
    """
    Tests whether, when the cache for an annotation is missing,
    sentences are labeled correctly.
    """
    sentence1 = (('I', 'Noun'), ('enjoy', 'Verb'), ('pizza', 'Noun'))
    token_annotation_sentence_1 = [1, 2, 3]
    pos_annotation_sentence_1 = [1, 2, 1]

    sentence2 = (('It', 'Det'), ('is', 'Verb'), ('tasty', 'Adj'))
    token_annotation_sentence_2 = [4, 5, 6]
    pos_annotation_sentence_2 = [3, 2, 4]

    args = {'device':'cpu'}
    input_fields = ['Token', 'PoS']

    train_path = self.tmpfile1.name
    dev_path = self.tmpfile2.name
    test_path = self.tmpfile3.name
    cache_model = cache.WholeDatasetCache(train_path, dev_path, test_path)

    token_task = task.TokenClassificationTask(args, 'Token', input_fields, cache_model)
    self.assertIsNone(token_task.train_cache)
    self.assert_annotations_equal(token_annotation_sentence_1, token_task.labels_of_sentence(sentence1, TRAIN_STR))
    self.assert_annotations_equal(token_annotation_sentence_2, token_task.labels_of_sentence(sentence2, TRAIN_STR))

    pos_task = task.TokenClassificationTask(args, 'PoS', input_fields, cache_model)
    self.assertIsNone(pos_task.train_cache)
    self.assert_annotations_equal(pos_annotation_sentence_1, pos_task.labels_of_sentence(sentence1, DEV_STR))
    self.assert_annotations_equal(pos_annotation_sentence_2, pos_task.labels_of_sentence(sentence2, DEV_STR))

  def test_use_cache_labels_of_sentence(self):
    """
    Tests whether, when the cache for an annotation is missing,
    sentences are labeled correctly.
    """
    sentence1 = (('I', 'Noun'), ('enjoy', 'Verb'), ('pizza', 'Noun'))
    token_annotation_sentence_1 = [1, 2, 3]
    pos_annotation_sentence_1 = [1, 2, 1]

    sentence2 = (('It', 'Det'), ('is', 'Verb'), ('tasty', 'Adj'))
    token_annotation_sentence_2 = [4, 5, 6]
    pos_annotation_sentence_2 = [3, 2, 4]

    args = {'device':'cpu'}
    input_fields = ['Token', 'PoS']

    train_path = self.tmpfile1.name
    dev_path = self.tmpfile2.name
    test_path = self.tmpfile3.name
    
    # Write the cache
    cache_model = cache.WholeDatasetCache(train_path, dev_path, test_path)

    token_task = task.TokenClassificationTask(args, 'Token', input_fields, cache_model)
    self.assert_annotations_equal(token_annotation_sentence_1, token_task.labels_of_sentence(sentence1, TRAIN_STR))
    self.assert_annotations_equal(token_annotation_sentence_2, token_task.labels_of_sentence(sentence2, TRAIN_STR))

    pos_task = task.TokenClassificationTask(args, 'PoS', input_fields, cache_model)
    self.assert_annotations_equal(pos_annotation_sentence_1, pos_task.labels_of_sentence(sentence1, DEV_STR))
    self.assert_annotations_equal(pos_annotation_sentence_2, pos_task.labels_of_sentence(sentence2, DEV_STR))

    # Use the cache
    cache_model = cache.WholeDatasetCache(train_path, dev_path, test_path)

    # Assert the caches still exist
    self.assertTrue(os.path.exists(cache_model.get_cache_path(TRAIN_STR, 'Token')))
    self.assertTrue(os.path.exists(cache_model.get_cache_path(TRAIN_STR, 'Token')))
    self.assertTrue(os.path.exists(cache_model.get_cache_path(DEV_STR, 'PoS')))
    self.assertTrue(os.path.exists(cache_model.get_cache_path(DEV_STR, 'PoS')))

    token_task = task.TokenClassificationTask(args, 'Token', input_fields, cache_model)
    self.assert_annotations_equal(token_annotation_sentence_1, token_task.labels_of_sentence(sentence1, TRAIN_STR))
    self.assert_annotations_equal(token_annotation_sentence_2, token_task.labels_of_sentence(sentence2, TRAIN_STR))
    self.assertIsNotNone(token_task.train_cache)

    pos_task = task.TokenClassificationTask(args, 'PoS', input_fields, cache_model)
    self.assert_annotations_equal(pos_annotation_sentence_1, pos_task.labels_of_sentence(sentence1, DEV_STR))
    self.assert_annotations_equal(pos_annotation_sentence_2, pos_task.labels_of_sentence(sentence2, DEV_STR))
    self.assertIsNotNone(pos_task.dev_cache)

class TestListDataset(DataTest):

  def setUp(self):
    """
    Writes temporary files with simple Ontonotes-formatted data
    """
    self.tmpfile1 = tempfile.NamedTemporaryFile(mode='w')
    self.tmpfile1.write("""I\tNoun\nenjoy\tVerb\npizza\tNoun""")
    self.tmpfile1.flush()
    self.tmpfile2 = tempfile.NamedTemporaryFile(mode='w')
    self.tmpfile2.write("""I\tNoun\nenjoy\tVerb\npizza\tNoun\n\nIt\tDet\nis\tVerb\ntasty\tAdj\n\n""")
    self.tmpfile2.flush()
    self.tmpfile3 = tempfile.NamedTemporaryFile(mode='w')
    self.tmpfile3.write("""Good\tAdj\n\nThis\tDet\nis\tVerb\na\tDet\nsentence\tNoun""")
    self.tmpfile3.flush()

  def tearDown(self):
    for cache_filename in glob.glob(self.tmpfile1.name + '.cache.*'):
      os.remove(cache_filename)
    self.tmpfile1.close()
    for cache_filename in glob.glob(self.tmpfile2.name + '.cache.*'):
      os.remove(cache_filename)
    self.tmpfile2.close()
    for cache_filename in glob.glob(self.tmpfile3.name + '.cache.*'):
      os.remove(cache_filename)
    self.tmpfile3.close()

  def test_single_annotation_input(self):
    """
    Tests spec of a list dataset with a single annotation
    input dataset.
    """
    args = {'device':'cpu'}
    input_fields = ['Token', 'PoS']

    train_path = self.tmpfile1.name
    dev_path = self.tmpfile2.name
    test_path = self.tmpfile3.name
    
    cache_model = cache.WholeDatasetCache(train_path, dev_path, test_path)

    data_reader = dataset.OntonotesReader(None, self.tmpfile1.name, None, None, None)

    sentence1 = (('I', 'Noun'), ('enjoy', 'Verb'), ('pizza', 'Noun'))
    token_annotation_sentence_1 = [1, 2, 3]
    pos_annotation_sentence_1 = [1, 2, 1]
    token_task = task.TokenClassificationTask(args, 'Token', input_fields, cache_model)
    pos_task = task.TokenClassificationTask(args, 'PoS', input_fields, cache_model)

    input_dataset = dataset.AnnotationData(args, token_task)
    output_dataset = dataset.AnnotationData(args, pos_task)

    list_dataset = dataset.ListDataset(args, data_reader, output_dataset, [input_dataset])
    data = next(list_dataset.load_data(TRAIN_STR))

    input_1  = data[0][0][0]
    self.assert_annotations_equal(token_annotation_sentence_1, input_1)

    output = data[1][0]
    self.assert_annotations_equal(pos_annotation_sentence_1, output)


  def test_two_annotation_input(self):
    """
    Tests spec of a list dataset with two annotation
    input dataset.
    """
    args = {'device':'cpu'}
    input_fields = ['Token', 'PoS']

    train_path = self.tmpfile1.name
    dev_path = self.tmpfile2.name
    test_path = self.tmpfile3.name
    
    cache_model = cache.WholeDatasetCache(train_path, dev_path, test_path)

    data_reader = dataset.OntonotesReader(None, self.tmpfile1.name, None, None, None)

    sentence1 = (('GI', 'Noun'), ('enjoy', 'Verb'), ('pizza', 'Noun'))
    token_annotation_sentence_1 = [1, 2, 3]
    pos_annotation_sentence_1 = [1, 2, 1]
    token_task = task.TokenClassificationTask(args, 'Token', input_fields, cache_model)
    pos_task = task.TokenClassificationTask(args, 'PoS', input_fields, cache_model)
    pos_task2 = task.TokenClassificationTask(args, 'PoS', input_fields)

    input_dataset1 = dataset.AnnotationData(args, token_task)
    input_dataset2 = dataset.AnnotationData(args, pos_task)
    output_dataset = dataset.AnnotationData(args, pos_task2)

    list_dataset = dataset.ListDataset(args, data_reader, output_dataset, [input_dataset1, input_dataset2])
    data = next(list_dataset.load_data(TRAIN_STR))

    input_1  = data[0][0][0]
    self.assert_annotations_equal(token_annotation_sentence_1, input_1)

    input_2  = data[0][1][0]
    self.assert_annotations_equal(pos_annotation_sentence_1, input_2)

    output = data[1][0]
    self.assert_annotations_equal(pos_annotation_sentence_1, output)

    input_dataset1 = dataset.AnnotationData(args, token_task)

  def test_dataloader_and_collate_fn(self):
    args = {'device':'cpu'}
    input_fields = ['Token', 'PoS']

    train_path = self.tmpfile1.name
    dev_path = self.tmpfile2.name
    test_path = self.tmpfile3.name

    data_reader = dataset.OntonotesReader(None, self.tmpfile1.name, self.tmpfile2.name, self.tmpfile3.name, None)

    token_task = task.TokenClassificationTask(args, 'Token', input_fields)
    pos_task = task.TokenClassificationTask(args, 'PoS', input_fields)
    pos_task2 = task.TokenClassificationTask(args, 'PoS', input_fields)

    input_dataset1 = dataset.AnnotationData(args, token_task)
    input_dataset2 = dataset.AnnotationData(args, pos_task)
    output_dataset = dataset.AnnotationData(args, pos_task2)

    list_dataset = dataset.ListDataset(args, data_reader, output_dataset, [input_dataset1, input_dataset2])

    train_dataloader = list_dataset.get_test_dataloader(shuffle=False)
    batch = next(iter(train_dataloader))
    ((inp_an, inpu_al), out_an, sent) = batch
    self.assert_2darray_equal(torch.tensor([[1., 0., 0., 0.], [2., 3., 4., 5.]]), inp_an[0]) 
    self.assert_2darray_equal(torch.tensor([[1., 0., 0., 0.], [2., 3., 2., 4.]]), inp_an[1]) 
    self.assert_2darray_equal(torch.tensor([[1., 0., 0., 0.], [2., 3., 2., 4.]]), out_an)
    self.assertEqual([[['Good', 'Adj']], [['This', 'Det'], ['is', 'Verb'], ['a', 'Det'], ['sentence', 'Noun']]], sent)

class TestAnnotationDataset(DataTest):

  def test_annotation_alignment(self):
    """
    Tests whether the alignment of annotation tensors to corpus-given tokens is
    the identity alignment (and checks the annotation)
    """
    args = {'device': 'cpu'}
    input_fields = ['Token', 'PoS']
    sentence1 = (('GI', 'Noun'), ('enjoy', 'Verb'), ('pizza', 'Noun'))
    pos_task = task.TokenClassificationTask(args, 'PoS', input_fields)
    input_dataset = dataset.AnnotationData(args, pos_task)
    annotation, alignment = input_dataset.tensor_of_sentence(sentence1, TRAIN_STR)
    print(annotation)
    print(alignment)
    self.assert_annotations_equal([1,2,1], annotation)
    self.assert_2darray_equal([[1,0,0],[0,1,0],[0,0,1]], alignment)

class TestHuggingfaceDataset(DataTest):

  def test_one_sentence_input(self):
    """
    Tests tokenization of a single sentence,
    and alignment between the tokenization and the corpus-given tokenization
    """
    model_string = "google/bert_uncased_L-2_H-128_A-2"
    args = {'device': 'cpu'}
    dataset_model = dataset.HuggingfaceData(args, model_string)

    sentence1 = (('0', 'Platypusbears', 'Noun'), ('1', 'eat', 'Verb'), ('2', 'pizza', 'Noun'))

    wordpiece_indices, wordpiece_alignment_vecs = dataset_model.tensor_of_sentence(sentence1, TRAIN_STR)

    self.assert_annotations_equal([101, 20228, 4017, 22571, 2271, 4783, 11650, 4521, 10733, 102],
        wordpiece_indices)
    self.assertEqual("[CLS] platypusbears eat pizza [SEP]", dataset_model.tokenizer.decode(wordpiece_indices))

    np.testing.assert_allclose(wordpiece_alignment_vecs, 
				[[0.0000, 0.0000, 0.0000],
        [0.166666667, 0.0000, 0.0000],
        [0.166666667, 0.0000, 0.0000],
        [0.166666667, 0.0000, 0.0000],
        [0.166666667, 0.0000, 0.0000],
        [0.166666667, 0.0000, 0.0000],
        [0.166666667, 0.0000, 0.0000],
        [0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [0.0000, 0.0000, 0.0000]])

class AnnotationModelTest(DataTest):

  def test_forward_wellformedness(self):
    """
    Tests whether the outout of the forward function
    of AnnotationModel is the one-hot representation
    of the indices passed to it
    """
    max_cardinality = 4
    trainable = False
    args = {'device':'cpu'}
    annotation_model = model.AnnotationModel(args, max_cardinality, trainable)

    annotation_tensor = torch.tensor(
        [[1,2,1,3,0],[3,3,3,3,3], [2,1,0,0,0]]
        )
    alignment_tensor = None
    output = annotation_model((annotation_tensor, alignment_tensor))

    batch_size = 3
    seq_len = 5
    self.assertEqual(output.shape, (batch_size, seq_len, max_cardinality))
    self.assert_annotations_equal([0,1,0,0], output[0,0])
    self.assert_annotations_equal([0,0,0,1], output[1,4])

  def test_pad_token_zeroed(self):
    """
    Tests whether the pad token (0) is given the zero vector,
    unlike all other token indices
    """
    max_cardinality = 4
    trainable = False
    args = {'device':'cpu'}
    annotation_model = model.AnnotationModel(args, max_cardinality, trainable)

    annotation_tensor = torch.tensor(
        [[1,2,1,3,0],[3,3,3,3,3], [2,1,0,0,0]]
        )
    alignment_tensor = None
    output = annotation_model((annotation_tensor, alignment_tensor))

    self.assert_annotations_equal([0,0,0,0], output[0,4])
    self.assert_annotations_equal([0,0,0,0], output[2,2])

  def test_trainable_is_respected(self):
    """
    Tests whether the trainable flag, which is True if
    a models should be fine-tuned during training, and False
    otherwise, is respected.

    That is, the embedding weights are trained (or not trained)
    depending on the boolean value of trainable
    """
    max_cardinality = 4
    args = {'device':'cpu'}
    for trainable in (False, True):
      annotation_model = model.AnnotationModel(args, max_cardinality, trainable)
      optimizer = torch.optim.SGD(annotation_model.embeddings.parameters(), lr=0.01)
      weights = torch.nn.Parameter(torch.zeros(max_cardinality))

      annotation_tensor = torch.tensor(
          [[1,2,1,3,0],[3,3,3,3,3], [2,1,0,0,0]]
          )
      alignment_tensor = None
      output = annotation_model((annotation_tensor, alignment_tensor))
      nn.init.uniform_(weights)

      # First pass
      prediction = torch.dot(weights, output[0,0])
      loss = (prediction - torch.tensor(1))**2
      loss.backward()
      optimizer.step()

      # Second pass
      new_output = annotation_model((annotation_tensor, alignment_tensor))
      new_prediction = torch.dot(weights, new_output[0,0])
      new_loss = (new_prediction - torch.tensor(1))**2

      if not trainable:
        self.assert_annotations_equal([0,1,0,0], annotation_model.embeddings.weight.data[1,:])
        self.assertAlmostEqual(new_loss, loss, places=7)
      else:
        self.assertLess(new_loss, loss)

class ListDatasetTest(DataTest): 

  def test_collate_fn(self):
    """
    Tests whether the outout of the forward function
    of AnnotationModel is the one-hot representation
    of the indices passed to it
    """
    model_string = "google/bert_uncased_L-2_H-128_A-2"
    args = {'device':'cpu'}
    dataset_model = dataset.HuggingfaceData(args, model_string)

    sentence1 = (('0', 'Platypusbears', 'Noun'), ('1', 'eat', 'Verb'), ('2', 'pizza', 'Noun'))
    sentence2 = (('0', 'They', 'Noun'), ('1', 'defenestrate', 'Verb'), ('2', 'ideologies', 'Noun'), ('3', 'precisely', 'Adv'))

    wordpiece_indices1, wordpiece_alignment_vecs1 = dataset_model.tensor_of_sentence(sentence1, TRAIN_STR)
    wordpiece_indices2, wordpiece_alignment_vecs2 = dataset_model.tensor_of_sentence(sentence2, TRAIN_STR)

    observation1 = ([(wordpiece_indices1, wordpiece_alignment_vecs1)],
        (wordpiece_indices1, wordpiece_alignment_vecs1), sentence1)
    observation2 = ([(wordpiece_indices2, wordpiece_alignment_vecs2)],
        (wordpiece_indices2, wordpiece_alignment_vecs2), sentence2)

    batch = dataset.ListDataset({'device':'cpu'}, None, None, None).collate_fn((observation1, observation2))
    input_batch, output_batch, sentences_batch = batch
    first_dataset_annotation_batch = input_batch[0][0]
    first_dataset_alignment_batch = input_batch[1][0]

    self.assert_2darray_equal(torch.tensor([[  101., 20228.,  4017., 22571.,  2271.,  4783., 11650.,  4521., 10733.,
           102.,     0.],
        [  101.,  2027., 13366., 28553.,  6494.,  2618.,  8909.,  8780., 21615.,
         10785.,   102.]]), output_batch)

    self.assert_2darray_equal(torch.tensor([[  101., 20228.,  4017., 22571.,  2271.,  4783., 11650.,  4521., 10733.,
           102.,     0.],
        [  101.,  2027., 13366., 28553.,  6494.,  2618.,  8909.,  8780., 21615.,
         10785.,   102.]]), first_dataset_annotation_batch)

    np.testing.assert_allclose(torch.tensor([[[0.0000, 0.0000, 0.0000, 0.0000],
         [0.1666666667, 0.0000, 0.0000, 0.0000],
         [0.1666666667, 0.0000, 0.0000, 0.0000],
         [0.1666666667, 0.0000, 0.0000, 0.0000],
         [0.1666666667, 0.0000, 0.0000, 0.0000],
         [0.1666666667, 0.0000, 0.0000, 0.0000],
         [0.1666666667, 0.0000, 0.0000, 0.0000],
         [0.0000, 1.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 1.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000]],

        [[0.0000, 0.0000, 0.0000, 0.0000],
         [1.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.2500, 0.0000, 0.0000],
         [0.0000, 0.2500, 0.0000, 0.0000],
         [0.0000, 0.2500, 0.0000, 0.0000],
         [0.0000, 0.2500, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.3333333333, 0.0000],
         [0.0000, 0.0000, 0.3333333333, 0.0000],
         [0.0000, 0.0000, 0.3333333333, 0.0000],
         [0.0000, 0.0000, 0.0000, 1.0000],
         [0.0000, 0.0000, 0.0000, 0.0000]]])
				, first_dataset_alignment_batch, atol=1e-6)

class HuggingfaceModelTest(DataTest):

  def test_forward_wellformedness(self):
    """
    Tests whether the outout of the forward function
    of HuggingfaceModel is computed er... at least
    in terms of the dimensions, and
    the alignment to token indices properly applied
    """

    max_cardinality = 4
    trainable = False
    layer_index = 2
    model_string = "google/bert_uncased_L-2_H-128_A-2"
    args = {'device':'cpu'}
    huggingface_model = model.HuggingfaceModel(args, model_string, trainable, layer_index)
    dataset_model = dataset.HuggingfaceData(args, model_string)

    sentence1 = (('0', 'Platypusbears', 'Noun'), ('1', 'eat', 'Verb'), ('2', 'pizza', 'Noun'))
    sentence2 = (('0', 'They', 'Noun'), ('1', 'defenestrate', 'Verb'), ('2', 'ideologies', 'Noun'), ('3', 'precisely', 'Adv'))

    wordpiece_indices1, wordpiece_alignment_vecs1 = dataset_model.tensor_of_sentence(sentence1, TRAIN_STR)
    wordpiece_indices2, wordpiece_alignment_vecs2 = dataset_model.tensor_of_sentence(sentence2, TRAIN_STR)

    observation1 = ([(wordpiece_indices1, wordpiece_alignment_vecs1)],
        (wordpiece_indices1, wordpiece_alignment_vecs1), sentence1)
    observation2 = ([(wordpiece_indices2, wordpiece_alignment_vecs2)],
        (wordpiece_indices2, wordpiece_alignment_vecs2), sentence2)

    batch = dataset.ListDataset({'device':'cpu'}, None, None, None).collate_fn((observation1, observation2))
    input_batch, output_batch, sentences_batch = batch

    first_dataset_annotation_batch = input_batch[0][0]
    first_dataset_alignment_batch = input_batch[1][0]

    huggingface_model_output = huggingface_model((first_dataset_annotation_batch, first_dataset_alignment_batch))
    self.assertEqual((2,4,128), huggingface_model_output.shape)

  def test_trainable_is_respected(self):
    """
    Tests whether the trainable flag, which is True if
    a models should be fine-tuned during training, and False
    otherwise, is respected.

    That is, the huggingface model are trained (or not trained)
    depending on the boolean value of trainable
    """
    for trainable in (False, True):
      layer_index = 2
      model_string = "google/bert_uncased_L-2_H-128_A-2"
      args = {'device':'cpu'}
      huggingface_model = model.HuggingfaceModel(args, model_string, trainable, layer_index)
      dataset_model = dataset.HuggingfaceData(args, model_string)

      sentence1 = (('0', 'Platypusbears', 'Noun'), ('1', 'eat', 'Verb'), ('2', 'pizza', 'Noun'))
      sentence2 = (('0', 'They', 'Noun'), ('1', 'defenestrate', 'Verb'), ('2', 'ideologies', 'Noun'), ('3', 'precisely', 'Adv'))

      wordpiece_indices1, wordpiece_alignment_vecs1 = dataset_model.tensor_of_sentence(sentence1, TRAIN_STR)
      wordpiece_indices2, wordpiece_alignment_vecs2 = dataset_model.tensor_of_sentence(sentence2, TRAIN_STR)

      observation1 = ([(wordpiece_indices1, wordpiece_alignment_vecs1)],
          (wordpiece_indices1, wordpiece_alignment_vecs1), sentence1)
      observation2 = ([(wordpiece_indices2, wordpiece_alignment_vecs2)],
          (wordpiece_indices2, wordpiece_alignment_vecs2), sentence2)

      batch = dataset.ListDataset(args, None, None, None).collate_fn((observation1, observation2))
      input_batch, output_batch, sentences_batch = batch

      first_dataset_annotation_batch = input_batch[0][0]
      first_dataset_alignment_batch = input_batch[1][0]

      huggingface_model_output = huggingface_model((first_dataset_annotation_batch, first_dataset_alignment_batch))

      # Optimizer and weights
      optimizer = torch.optim.SGD(huggingface_model.parameters(), lr=0.000000001)
      weights = torch.nn.Parameter(torch.zeros(huggingface_model_output.shape[2]))
      nn.init.uniform_(weights)

      # First pass
      prediction = torch.dot(weights, huggingface_model_output[0,0])
      loss = (prediction - torch.tensor(1))**2
      loss.backward()
      optimizer.step()

      # Second pass
      new_output = huggingface_model((first_dataset_annotation_batch, first_dataset_alignment_batch))
      new_prediction = torch.dot(weights, new_output[0,0])
      new_loss = (new_prediction - torch.tensor(1))**2

      if not trainable:
        self.assertAlmostEqual(new_loss, loss, places=7)
      else:
        self.assertLess(new_loss, loss)

class ListModelTest(DataTest):

  def setUp(self):
    """
    Writes temporary files with simple Ontonotes-formatted data
    """
    self.tmpfile1 = tempfile.NamedTemporaryFile(mode='w')
    self.tmpfile1.write('\n'.join(
      ( '\t'.join(x) for x in
        (('0', 'Platypusbears', 'Noun'), ('1', 'eat', 'Verb'), ('2', 'pizza', 'Noun')))))
    self.tmpfile1.write('\n\n')
    self.tmpfile1.write('\n'.join(
      ( '\t'.join(x) for x in
        (('0', 'They', 'Noun'), ('1', 'defenestrate', 'Verb'), ('2', 'ideologies', 'Noun'), ('3', 'precisely', 'Adv')))))
    self.tmpfile1.flush()

  def tearDown(self):
    self.tmpfile1.close()

  def test_annotation_and_huggingface_list(self):
    """
    Tests whether the outout of the forward function
    of ListModel computes each representation and
    concatenates their results.
    """
    args = {'device':'cpu'}
    input_fields = ['Token', 'PoS']
    max_cardinality = 4
    trainable = False
    layer_index = 2
    model_string = "google/bert_uncased_L-2_H-128_A-2"
    huggingface_dataset = dataset.HuggingfaceData(args, model_string)

    pos_task = task.TokenClassificationTask(args, 'PoS', input_fields)
    pos_dataset = dataset.AnnotationData(args, pos_task)
    token_task = task.TokenClassificationTask(args, 'Token', input_fields)
    token_dataset = dataset.AnnotationData(args, token_task)

    generator = dataset.OntonotesReader(None, self.tmpfile1.name, None, None, None)

    list_dataset = dataset.ListDataset(args, generator, pos_dataset,
        [token_dataset, huggingface_dataset])

    annotation_model = model.AnnotationModel(args, max_cardinality=10, trainable=False)
    huggingface_model = model.HuggingfaceModel(args, model_string, trainable, layer_index)
    list_model = model.ListModel(args, [annotation_model, huggingface_model])

    train_dataloader = list_dataset.get_train_dataloader(shuffle=False)
    batch = next(iter(train_dataloader))

    inputs, outputs, sentences = batch
    output = list_model(inputs)
    self.assertEqual((2,4,128+10), output.shape)
    self.assert_annotations_equal([0.0000,   1.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
      0.0000, 0.0000, 0.0000], output[0,0,:10])

  def test_annotation_and_two_huggingface_model_list(self):
    """
    Tests whether the outout of the forward function
    of ListModel computes each representation and
    concatenates their results.
    """
    args = {'device':'cpu'}
    input_fields = ['Token', 'PoS']
    max_cardinality = 4
    trainable = False
    layer_index = 2
    model_string1 = "google/bert_uncased_L-2_H-128_A-2"
    model_string2 = "google/bert_uncased_L-4_H-128_A-2"
    huggingface_dataset1 = dataset.HuggingfaceData(args, model_string1)
    huggingface_dataset2 = dataset.HuggingfaceData(args, model_string2)

    pos_task = task.TokenClassificationTask(args, 'PoS', input_fields)
    pos_dataset = dataset.AnnotationData(args, pos_task)
    token_task = task.TokenClassificationTask(args, 'Token', input_fields)
    token_dataset = dataset.AnnotationData(args, token_task)

    generator = dataset.OntonotesReader(None, self.tmpfile1.name, None, None, None)

    list_dataset = dataset.ListDataset(args, generator, pos_dataset,
        [token_dataset, huggingface_dataset1, huggingface_dataset2])

    annotation_model = model.AnnotationModel(args, max_cardinality=10, trainable=False)
    huggingface_model1 = model.HuggingfaceModel(args, model_string1, trainable, layer_index)
    huggingface_model2 = model.HuggingfaceModel(args, model_string2, trainable, layer_index)
    list_model = model.ListModel(args, [annotation_model, huggingface_model1, huggingface_model2])

    train_dataloader = list_dataset.get_train_dataloader(shuffle=False)
    batch = next(iter(train_dataloader))

    inputs, outputs, sentences = batch
    output = list_model(inputs)
    self.assertEqual((2,4,128+128+10), output.shape)
    self.assert_annotations_equal([0.0000,   1.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
      0.0000, 0.0000, 0.0000], output[0,0,:10])

class LinearTokenLabelProbeTest(DataTest):

    def test_wellformedness(self):
        """
        Tests whether the probe forward function computes
        a function that transforms tensors of shape
        (batch_size, seq_len, feature_count)
        to tensors of shape
        (batch_size, seq_len, output_size)
        """

        args = {'device':'cpu'}
        model_dim = 10
        linear_label_probe = probe.OneWordLinearLabelProbe(
                args, 128, model_dim)

        batch_size, seq_len, feature_count = 4, 50, 128
        batch = torch.ones((batch_size, seq_len, feature_count))

        output = linear_label_probe(batch)
        self.assertEqual((4,50,model_dim), output.shape)

class TrainerTest(DataTest):

  def setUp(self):
    """
    Writes temporary files with simple Ontonotes-formatted data
    """
    self.tmpfile1 = tempfile.NamedTemporaryFile(mode='w')
    self.tmpfile1.write('\n'.join(
      ( '\t'.join(x) for x in
        (('0', 'Platypusbears', 'Noun'), ('1', 'eat', 'Verb'), ('2', 'pizza', 'Noun')))))
    self.tmpfile1.write('\n\n')
    self.tmpfile1.write('\n'.join(
      ( '\t'.join(x) for x in
        (('0', 'They', 'Noun'), ('1', 'defenestrate', 'Verb'), ('2', 'ideologies', 'Noun'), ('3', 'precisely', 'Adv')))))
    self.tmpfile1.flush()

  def tearDown(self):
    self.tmpfile1.close()
    os.remove('.params')
  
  def test_wellformedness(self):
    args = {'device':'cpu'}
    input_fields = ['Token', 'PoS']
    reporting_root = '.'

    # Data loaders and feeders
    pos_task = task.TokenClassificationTask(args, 'PoS', input_fields)
    pos_dataset = dataset.AnnotationData(args, pos_task)
    token_task = task.TokenClassificationTask(args, 'Token', input_fields)
    token_dataset = dataset.AnnotationData(args, token_task)

    generator = dataset.OntonotesReader(None, self.tmpfile1.name, None, None, None)

    list_dataset = dataset.ListDataset(args, generator, pos_dataset,
        [token_dataset])

    train_dataloader = list_dataset.get_train_dataloader(shuffle=False)
    dev_dataloader = list_dataset.get_train_dataloader(shuffle=False)

    # Models for generating representations
    token_model = model.AnnotationModel(args, max_cardinality=10, trainable=True)
    list_model = model.ListModel(args, [token_model])
  
    # The probe model
    pos_dim = 5
    linear_label_probe = probe.OneWordLinearLabelProbe(
            args, 10, 10)

    # Training procedure
    regimen = trainer.ProbeRegimen(args, max_epochs=2, params_path='.params', reporting_root=reporting_root)

    dev_losses = regimen.train_until_convergence(linear_label_probe, list_model, None, train_dataloader, dev_dataloader,
        gradient_steps_between_eval=1)
    self.assertLess(dev_losses[1], dev_losses[0])

class LabelReporterTest(DataTest):
  """
  Tests reporting of metrics for single-token tasks
  """

  def setUp(self):
    """
    Writes temporary files with simple Ontonotes-formatted data
    """
    self.tmpfile1 = tempfile.NamedTemporaryFile(mode='w')
    self.tmpfile1.write('\n'.join(
      ( '\t'.join(x) for x in
        (('0', 'Platypusbears', 'Noun'), ('1', 'eat', 'Verb'), ('2', 'pizza', 'Noun')))))
    self.tmpfile1.write('\n\n')
    self.tmpfile1.write('\n'.join(
      ( '\t'.join(x) for x in
        (('0', 'They', 'Noun'), ('1', 'defenestrate', 'Verb'), ('2', 'ideologies', 'Noun'), ('3', 'precisely', 'Adv')))))
    self.tmpfile1.flush()

  def tearDown(self):
    self.tmpfile1.close()
    os.remove('train.label_acc')

  def test_label_accuracy_v_entropy(self):
    args = {'device':'cpu'}
    reporting_root = '.'
    input_fields = ['Token', 'PoS']

    batch_size, seq_len, label_count = 2, 4, 8
    predictions = torch.zeros(batch_size, seq_len, label_count)
    predictions[0,0,1], predictions[0,1,2], predictions[0,2,3] = 1,1,1
    predictions[1,0,1], predictions[1,1,2], predictions[1,2,3], predictions[1,3,0] = 1,1,1,1
    #predictions = predictions.cpu().numpy()
    #labels = torch.tensor([[1,2,3],[2,3,0]])

    #
    #args = {'input_fields': ['Token', 'PoS'], "reporting":{"root":"."}}

    # Data loaders and feeders
    pos_task = task.TokenClassificationTask(args, 'PoS', input_fields)
    pos_dataset = dataset.AnnotationData(args, pos_task)
    token_task = task.TokenClassificationTask(args, 'Token', input_fields)
    token_dataset = dataset.AnnotationData(args, token_task)

    generator = dataset.OntonotesReader(None, self.tmpfile1.name, None, None, None)

    list_dataset = dataset.ListDataset(args, generator, pos_dataset,
        [token_dataset])

    train_dataloader = list_dataset.get_train_dataloader(shuffle=False)
    dev_dataloader = list_dataset.get_train_dataloader(shuffle=False)

    # The reporter
    reporting_methods = ['label_accuracy', 'v_entropy']
    reporter_class = reporter.IndependentLabelReporter(args, reporting_root, reporting_methods)
    reporter_class([predictions], train_dataloader, TRAIN_STR)
    with open('train.label_acc') as fin:
        acc = float(fin.read().strip())
        self.assertAlmostEqual(acc, 0.42857142857142855)
    with open('train.v_entropy') as fin:
        acc = float(fin.read().strip())
        self.assertAlmostEqual(acc, 1.8454373223440987)

ner_data = \
"""1\tparts\tpart\tNOUN\tNNS\t_\t0\troot\t_\t_\twb/eng/00/eng_0017\t5\t0\tparts\tNNS\t(TOP(NP(NP*)\tpart\t-\t-\t_lekker_&lt;lek...@intergate.bc.ca&gt;_\t*\t*\t-
3\tThe\tthe\tDET\tDT\t_\t5\tdet\t_\t_\twb/eng/00/eng_0017\t5\t2\tThe\tDT\t(NP(NP(NP*\t-\t-\t-\t_lekker_&lt;lek...@intergate.bc.ca&gt;_\t(WORK_OF_ART*\t*\t-
4\tBurning\tBurning\tPROPN\tNNP\t_\t5\tcompound\t_\t_\twb/eng/00/eng_0017\t5\t3\tBurning\tNNP\t*\t-\t-\t-\t_lekker_&lt;lek...@intergate.bc.ca&gt;_\t*\t*\t-
5\tRoadblocks\tRoadblocks\tPROPN\tNNPS\t_\t1\tnmod\t_\t_\twb/eng/00/eng_0017\t5\t4\tRoadblocks\tNNPS\t*)\t-\t-\t-\t_lekker_&lt;lek...@intergate.bc.ca&gt;_\t*)\t*\t-
1\tparts\tpart\tNOUN\tNNS\t_\t0\troot\t_\t_\twb/eng/00/eng_0017\t5\t0\tparts\tNNS\t(TOP(NP(NP*)\tpart\t-\t-\t_lekker_&lt;lek...@intergate.bc.ca&gt;_\t*\t*\t-
24\tRyszard\tRyszard\tPROPN\tNNP\t_\t25\tcompound\t_\t_\twb/eng/00/eng_0017\t5\t23\tRyszard\tNNP\t(NP*\t-\t-\t-\t_lekker_&lt;lek...@intergate.bc.ca&gt;_\t(PERSON*\t*\t-
25\tski\tski\tPROPN\tNNP\t_\t21\tnmod\t_\t_\twb/eng/00/eng_0017\t5\t24\tski\tNNP\t*))\t-\t-\t-\t_lekker_&lt;lek...@intergate.bc.ca&gt;_\t*)\t*\t-
31\t1991\t1991\tNUM\tCD\t_\t29\tamod\t_\t_\twb/eng/00/eng_0017\t5\t30\t1991\tCD\t(NP*))))))))\t-\t-\t-\t_lekker_&lt;lek...@intergate.bc.ca&gt;_\t(DATE)\t*\t-
8\tfor\tfor\tADP\tIN\t_\t10\tcase\t_\t_\twb/eng/00/eng_0017\t5\t7\tfor\tIN\t(PP*\t-\t-\t-\t_lekker_&lt;lek...@intergate.bc.ca&gt;_\t*\t*\t-
31\t1991\t1991\tNUM\tCD\t_\t29\tamod\t_\t_\twb/eng/00/eng_0017\t5\t30\t1991\tCD\t(NP*))))))))\t-\t-\t-\t_lekker_&lt;lek...@intergate.bc.ca&gt;_\t(DATE)\t*\t-
27\tAlfred\tAlfred\tPROPN\tNNP\t_\t29\tcompound\t_\t_\twb/eng/00/eng_0017\t5\t26\tAlfred\tNNP\t(NP(NP*\t-\t-\t-\t_lekker_&lt;lek...@intergate.bc.ca&gt;_\t(ORG*\t*\t-
28\tA.\tA.\tPROPN\tNNP\t_\t29\tcompound\t_\t_\twb/eng/00/eng_0017\t5\t27\tA.\tNNP\t*\t-\t-\t-\t_lekker_&lt;lek...@intergate.bc.ca&gt;_\t*\t*\t-
29\tKnopf\tKnopf\tPROPN\tNNP\t_\t21\tappos\t_\t_\twb/eng/00/eng_0017\t5\t28\tKnopf\tNNP\t*)\t-\t-\t-\t_lekker_&lt;lek...@intergate.bc.ca&gt;_\t*)\t*\t-"""

class NERTest(DataTest):
  """
  Tests loading and scoring of Named Entity Recognition tags
  """

  def setUp(self):
    """
    Writes temporary files with simple Ontonotes-formatted data
    """
    self.tmpfile1 = tempfile.NamedTemporaryFile(mode='w')
    self.tmpfile1.write(ner_data)
    self.tmpfile1.flush()

  def test_ner_bioes_tags(self):
    """
    Tests whether NER annotations are as expected, on simulated NER data
    The raw NER labels in the ontonotes look like this :
    (WORK_OF_ART

    )

    (PERSON
        )
    (DATE)

    (DATE)
    (ORG

        )
    """
    args = {}
    reporting_root = '.'
    input_fields = ['index', 'token', 'lemma', 'upos', 'xpos', '-', 'head_index', 'dep_rel', '-', '-', 'section', 'index2', 'xpos2', 'parse_bit', 'lemma','-','-','speaker','-', '-', 'named_entities','-']

    ner_task = task.NERClassificationTask(args, 'named_entities', input_fields)
    ner_task._manual_setup()
    sentence = [x.split('\t') for x in ner_data.split('\n')]
    string_labels = ner_task._string_labels_of_sentence(sentence)
    self.assert_annotations_equal(string_labels, ['O', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'E-WORK_OF_ART', 'O', 'B-PERSON', 'E-PERSON', 'S-DATE', 'O', 'S-DATE', 'B-ORG', 'I-ORG', 'E-ORG'])
    integer_labels = ner_task.labels_of_sentence(sentence, TRAIN_STR)
    np.testing.assert_allclose(integer_labels, [ 1.,  2.,  3.,  4.,  1.,  5.,  6.,  7.,  1.,  7.,  8., 9., 10.])

  def test_ner_f1_evaluation(self):
    args = {'device':'cpu'}
    reporting_root = '.'
    input_fields = ['index', 'token', 'lemma', 'upos', 'xpos', '-', 'head_index', 'dep_rel', '-', '-', 'section', 'index2', 'xpos2', 'parse_bit', 'lemma','-','-','speaker','-', '-', 'named_entities','-']

    ner_task = task.NERClassificationTask(args, 'named_entities', input_fields)
    ner_task._manual_setup()
    sentence = [x.split('\t') for x in ner_data.split('\n')]

    batch_size, seq_len, label_count = 1, 13, 11
    predictions = torch.zeros(batch_size, seq_len, label_count)
    #predictions[0,0,1], predictions[0,1,2], predictions[0,2,3] = 1,1,1
    predictions[0,0,1], predictions[0,1,3], predictions[0,2,4], predictions[0,3,2] = 1,1,1,1
    predictions[0,4,1], predictions[0,5,5], predictions[0,6,5], predictions[0,7,7] = 1,1,1,1
    predictions[0,8,10], predictions[0,9,7], predictions[0,10,8], predictions[0,11,9] = 1,1,1,1
    predictions[0,12,10] = 1

    ner_dataset = dataset.AnnotationData(args, ner_task)
    token_task = task.TokenClassificationTask(args, 'token', input_fields)
    token_dataset = dataset.AnnotationData(args, token_task)

    generator = dataset.OntonotesReader(None, self.tmpfile1.name, None, None, None)

    list_dataset = dataset.ListDataset(args, generator, ner_dataset,
        [token_dataset])

    train_dataloader = list_dataset.get_train_dataloader(shuffle=False)
    dev_dataloader = list_dataset.get_train_dataloader(shuffle=False)

    # The reporter
    reporting_methods = ['ner_f1', 'v_entropy']
    reporter_class = reporter.NERReporter(args, reporting_root, reporting_methods, ner_task)
    reporter_class([predictions], train_dataloader, TRAIN_STR)
    with open('train.precision') as fin:
        acc = float(fin.read().strip())
        self.assertAlmostEqual(acc, .375)
    with open('train.recall') as fin:
        acc = float(fin.read().strip())
        self.assertAlmostEqual(acc, .6)
    with open('train.f1') as fin:
        acc = float(fin.read().strip())
        self.assertAlmostEqual(acc, 0.4615384615384615)


if __name__ == '__main__':
  unittest.main()
