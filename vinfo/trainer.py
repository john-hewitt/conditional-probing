import os 
import sys
from yaml import YAMLObject

from utils import InitYAMLObject

from tqdm import tqdm
import torch
from torch import optim

from loss import CustomCrossEntropyLoss

class ProbeRegimen(InitYAMLObject):
  """Basic regimen for training and running inference on probes.
  
  Tutorial help from:
  https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

  Attributes:
    optimizer: the optimizer used to train the probe
    scheduler: the scheduler used to set the optimizer base learning rate
  """
  yaml_tag = '!ProbeRegimen'

  def __init__(self, args, max_epochs, params_path, reporting_root, max_gradient_steps=-1, eval_dev_every=-1):
    self.args = args
    self.max_epochs = max_epochs
    #self.params_path = os.path.join(args['reporting']['root'], params_path)
    self.reporting_root = reporting_root
    self.params_name = params_path
    self.max_gradient_steps = sys.maxsize if max_gradient_steps==-1 else max_gradient_steps
    self.eval_dev_every = eval_dev_every
    self.loss = CustomCrossEntropyLoss(args)

  def train_until_convergence(self, probe, model, loss, train_dataset, dev_dataset, gradient_steps_between_eval, finetune=False):
    """ Trains a probe until a convergence criterion is met.

    Trains until loss on the development set does not improve by more than epsilon
    for 5 straight epochs.

    Writes parameters of the probe to disk, at the location specified by config.

    Args:
      probe: An instance of probe.Probe, transforming model outputs to predictions
      model: An instance of model.Model, transforming inputs to word reprs
      loss: An instance of loss.Loss, computing loss between predictions and labels
      train_dataset: a torch.DataLoader object for iterating through training data
      dev_dataset: a torch.DataLoader object for iterating through dev data
    """
    self.params_path = os.path.join(self.reporting_root, self.params_name)
    if finetune:
      self.optimizer = optim.Adam(list(probe.parameters()) + list(model.parameters()), lr=0.00001, weight_decay=0)
    else:
      self.optimizer = optim.Adam(probe.parameters(), lr=0.001, weight_decay=0)
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,patience=0)
    #loss = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    min_dev_loss = sys.maxsize
    min_dev_loss_epoch = -1
    gradient_steps = 0
    #eval_dev_every = self.eval_dev_every if self.eval_dev_every != -1 else EVAL_EVERY
    eval_dev_every = gradient_steps_between_eval

    eval_index = 0
    min_dev_loss_eval_index = -1
    eval_dev_losses = []
    for epoch_index in tqdm(range(self.max_epochs), desc='[training]'):
      epoch_train_loss = 0
      epoch_train_loss_count = 0
      for batch in tqdm(train_dataset, desc='[training]'):
        # Take a train step
        probe.train()
        self.optimizer.zero_grad()
        input_batch, output_batch, sentences  = batch
        word_representations = model(input_batch)
        predictions = probe(word_representations)
        batch_loss, count = self.loss(predictions, output_batch)
        batch_loss.backward()
        epoch_train_loss += batch_loss.detach().cpu().numpy()
        epoch_train_loss_count += count.detach().cpu().numpy()
        self.optimizer.step()
        gradient_steps += 1
        if gradient_steps % eval_dev_every == 0:
          eval_index += 1
          if gradient_steps >= self.max_gradient_steps:
            tqdm.write('Hit max gradient steps; stopping')
            return eval_dev_losses
          epoch_dev_loss = 0
          epoch_dev_loss_count = 0
          for batch in tqdm(dev_dataset, desc='[dev batch]'):
            self.optimizer.zero_grad()
            probe.eval()
            input_batch, output_batch, _ = batch
            word_representations = model(input_batch)
            predictions = probe(word_representations)
            batch_loss, count = self.loss(predictions, output_batch)
            epoch_dev_loss += batch_loss.detach().cpu().numpy()
            epoch_dev_loss_count += count.detach().cpu().numpy()
          self.scheduler.step(epoch_dev_loss)
          tqdm.write('[epoch {}] Train loss: {}, Dev loss: {}'.format(epoch_index,
              epoch_train_loss/epoch_train_loss_count, epoch_dev_loss/epoch_dev_loss_count))
          eval_dev_losses.append(epoch_dev_loss)
          if epoch_dev_loss / epoch_dev_loss_count < min_dev_loss - 0.001:
            torch.save(probe.state_dict(), self.params_path)
            #torch.save(model.state_dict(), self.params_path + '.model')
            min_dev_loss = epoch_dev_loss / epoch_dev_loss_count
            min_dev_loss_epoch = epoch_index
            min_dev_loss_eval_index = eval_index
            tqdm.write('Saving probe parameters to {}'.format(self.params_path))
          elif min_dev_loss_eval_index <= eval_index - 3:
            tqdm.write('Early stopping')
            return eval_dev_losses
    return eval_dev_losses

  def predict(self, probe, model, dataset):
    """ Runs probe to compute predictions on a dataset.

    Args:
      probe: An instance of probe.Probe, transforming model outputs to predictions
      model: An instance of model.Model, transforming inputs to word reprs
      dataset: A pytorch.DataLoader object 

    Returns:
      A list of predictions for each batch in the batches yielded by the dataset
    """
    probe.eval()
    predictions_by_batch = []
    for batch in tqdm(dataset, desc='[predicting]'):
      input_batch, label_batch, _ = batch
      word_representations = model(input_batch)
      predictions = probe(word_representations)
      predictions_by_batch.append(predictions.detach().cpu())#.numpy())
    return predictions_by_batch

