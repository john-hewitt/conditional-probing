import yaml as yaml
import torch
import torch.nn as nn
import sys
import click
import os

import model
import dataset
import task
import cache
import probe
import trainer
import reporter
from utils import TRAIN_STR, DEV_STR, TEST_STR

ontonotes_fields = ["one_offset_word_index", "token", "None", "ptb_pos", "ptb_pos2", "None2", "dep_rel", "None3", "None4", "source_file", "part_number", "zero_offset_word_index", "token2", "ptb_pos3", "parse_bit", "predicate_lemma", "predicate_frameset_id", "word_sense", "speaker_author", "named_entities"]

from model    import * 
from dataset  import * 
from task     import * 
from cache    import * 
from probe    import * 
from trainer  import * 
from reporter import * 
from utils import *

@click.command()
@click.argument('yaml_path')
@click.option('--just-cache-data', default=0, help='If 1, just writes data to cache; does not run experiment')
@click.option('--do_test', default=0, help='If 1, evaluates on the test set; hopefully just run this once!')
def run_yaml_experiment(yaml_path, just_cache_data, do_test):
  """
  Runs an experiment as configured by a yaml config file
  """

  # Take constructed classes from yaml
  yaml_args = yaml.load(open(yaml_path), Loader=yaml.Loader)
  list_dataset = yaml_args['dataset']
  list_model = yaml_args['model']
  probe_model = yaml_args['probe']
  regimen_model = yaml_args['regimen']
  reporter_model = yaml_args['reporter']
  cache_model = yaml_args['cache']

  # Make results directory
  os.makedirs(regimen_model.reporting_root, exist_ok=True)

  # Make dataloaders and load data
  train_dataloader = list_dataset.get_train_dataloader(shuffle=True)
  dev_dataloader = list_dataset.get_dev_dataloader(shuffle=False)
  if do_test:
    test_dataloader = list_dataset.get_test_dataloader(shuffle=False)
  cache_model.release_locks()

  if just_cache_data:
    print("Data caching done. Exiting...")
    return

  # Train probe
  regimen_model.train_until_convergence(probe_model, list_model
      , None, train_dataloader, dev_dataloader
      , gradient_steps_between_eval=min(1000,len(train_dataloader)))

  # Train probe with finetuning
  #regimen_model.train_until_convergence(probe_model, list_model
  #    , None, train_dataloader, dev_dataloader
  #    , gradient_steps_between_eval=1000, finetune=True)

  # Load best probe from disk
  probe_model.load_state_dict(torch.load(regimen_model.params_path))
  #list_model.load_state_dict(torch.load(regimen_model.params_path + '.model'))

  # Make dataloaders and predict
  train_dataloader = list_dataset.get_train_dataloader(shuffle=False)
  dev_dataloader = list_dataset.get_dev_dataloader(shuffle=False)
  dev_predictions = regimen_model.predict(probe_model, list_model, dev_dataloader)
  train_predictions = regimen_model.predict(probe_model, list_model, train_dataloader)
  if do_test:
    test_dataloader = list_dataset.get_test_dataloader(shuffle=False)
    test_predictions = regimen_model.predict(probe_model, list_model, test_dataloader)
  
  # Make dataloaders and report
  train_dataloader = list_dataset.get_train_dataloader(shuffle=False)
  dev_dataloader = list_dataset.get_dev_dataloader(shuffle=False)
  reporter_model(train_predictions, train_dataloader, TRAIN_STR)
  reporter_model(dev_predictions, dev_dataloader, DEV_STR)
  if do_test:
    test_dataloader = list_dataset.get_test_dataloader(shuffle=False)
    reporter_model(test_predictions, test_dataloader, TEST_STR)

if __name__ == '__main__':
  run_yaml_experiment()
