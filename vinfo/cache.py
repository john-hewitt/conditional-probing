import os
from yaml import YAMLObject
import logging
import itertools
from utils import TRAIN_STR, DEV_STR, TEST_STR, InitYAMLObject
import glob
from pathlib import Path
import h5py

class WholeDatasetCache(InitYAMLObject):
  """
  Class for managing the storage and recall of
  precomputed featurized versions of datasets and annotations
  """
  cache_checked = False
  yaml_tag = '!WholeDatasetCache'

  def __init__(self, train_path, dev_path, test_path, force_read_cache=False):
    self.train_path = train_path
    self.dev_path = dev_path
    self.test_path = test_path
    self.force_read_cache = force_read_cache

  def get_cache_path_and_check(self, split, task_name):
    """Provides the path for cache files, and cache validity

    Arguments:
      split: {TRAIN_STR, DEV_STR, TEST_STR} determining data split
      task_name: unique identifier for task/annotation type
    Returns:
      - filesystem path for the cache
      - bool True if the cache is valid to be read from
        (== exists and no lock file exists indicating that it is
         being written to. Does not solve race conditions; use
         cache with caution.)
    """
    task_name = "".join([c for c in task_name if c.isalpha() or c.isdigit() or c==' ']).rstrip()
    if split == TRAIN_STR:
      read_cache, write_cache = self.check_cache(self.train_path, task_name)
      print('For task {}, split {}, we are reading:{}, writing:{} the cache'.format(task_name, split, read_cache, write_cache))
      return self.train_path + '.cache.' + task_name + '.hdf5', read_cache, write_cache
    elif split == DEV_STR:
      read_cache, write_cache = self.check_cache(self.dev_path, task_name)
      print('For task {}, split {}, we are reading:{}, writing:{} the cache'.format(task_name, split, read_cache, write_cache))
      return self.dev_path + '.cache.' + task_name + '.hdf5', read_cache, write_cache
    elif split == TEST_STR:
      read_cache, write_cache = self.check_cache(self.test_path, task_name)
      print('For task {}, split {}, we are reading:{}, writing:{} the cache'.format(task_name, split, read_cache, write_cache))
      return self.test_path + '.cache.' + task_name + '.hdf5', read_cache, write_cache
    else:
      return ValueError("Unknown split name: {}".format(split))


  def check_cache(self, path, task_name):
    """Determines whether datasets have changed; erases caches if so; checks cache lock

    At the path given, a dataset is required to be there; else an error
    is thrown. Each file at a path of the form ${path}*.cache is erased
    if it is older than the file at ${path}.
    Further, if in this process another object has already started to write
    to this cache, then use of the cache is disabled.

    Arguments:
      path: The full disk path to a dataset
    Outputs:
      (read_cache, write_cache):
        read_cache True if cache should be used for reading, False otherwise
        write_cache True if cache should be written to, False otherwise
    """
    task_name = "".join([c for c in task_name if c.isalpha() or c.isdigit() or c==' ']).rstrip()
    dataset_time = os.path.getmtime(path)
    if self.force_read_cache:
      print('Forcing trying to read cache, even if not there')
      return True, False # Force trying to read from the cache
    for cache_path in glob.glob(path + '*cache.' + task_name + '.*hdf5'):
      lock_path = cache_path + '.lock'
      cache_time = os.path.getmtime(cache_path)
      if cache_time < dataset_time:
        os.remove(cache_path)
        logging.info('Cache erased at: {}'.format(cache_path))
        if os.path.exists(lock_path):
          os.remove(lock_path)
        return False, True # Cache older than data; erased and written to
      if os.path.exists(lock_path):
        return False, False # Cache locked, being written to by object in this process
      return True, False # Cache is valid; read from it
    return False, True # No cache exists; write one


  def release_locks(self):
    """ Removes lock files from caches
    """
    lock_paths = itertools.chain(
        glob.glob(self.train_path + '*.lock'),
        glob.glob(self.dev_path + '*.lock'),
        glob.glob(self.test_path + '*.lock'))
    for cache_lock_path in lock_paths:
      print('Removing cache lock file at {}'.format(cache_lock_path))
      os.remove(cache_lock_path)

  def get_hdf5_cache_writer(self, cache_path):
    """ Gets an hdf5 file writer and makes a lock file for it
    """
    print('Getting cache writer for {}'.format(cache_path))
    Path(cache_path + '.lock').touch()
    return h5py.File(cache_path, 'a')



  #def is_valid():
  #  """
  #  Getter method for whether the data cached is valid to be used again

  #  Returns:
  #    True if Task models can reuse 
  #  """
  #  if not self.cache_checked:
  #    raise Exception("Cache has not been checked but is being queried")


    
