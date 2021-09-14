# conditional-probing

Codebase for easy specification of variational (conditional) information probe experiments.

Highlights:
 - **Conditional probing**: measure only the aspects of property that aren't explainable by the baseline of your choice.
 - Train a probe using one or many layers from one or many models as input representations; loading and concatenation of representations performed automatically.
 - Integration with huggingface for specifying representation.
 - Heuristic subword-token-to-token alignment of [Tenney et al., 2019](https://openreview.net/forum?id=SJzSgnRcKX) performed per-model.
 - Sort of smart caching of tokenized datasets and subword token alignment matrices to `hdf5` files.
 - Modular design of probes, training regimen, representation models, and reporting.
 - Change out classes specifying probes or representations directly through YAML configs instead of `if` statements in code.


Written for the paper [Conditional probes: measuring usable information beyond a baseline]().

## Installing and getting started
1. Clone the repository.

        git clone https://github.com/john-hewitt/vinfo-probing-internal/
        cd vinfo-probing-internal
        
1. [Optional] Construct a virtual environment for this project. Only `python3` is supported.

        conda create --name sp-env
        conda activate sp-env

1. Install the required packages. 

        conda install --file requirements.txt

1. Run your first experiment using a provided config file. This experiment trains and reports a part-of-speech probe on layer 5 of the `roberta-base` model.

        python vinfo/experiment.py example/roberta768-upos-layer5-example-cpu.yaml

1. Take a look at the config file, `example/roberta768-upos-layer5-example.yaml`. It states that the results and probe parameters are saved to `example/`, a directory that would've been created if it hadn't already existed. If your experiment ran without error, you should see the following files in that directory:

        dev.v_entropy
        train.v_entropy
        dev.label_acc
        train.label_acc
        params
       
    The `v_entropy` files store a single float: the variational entropy as estimated on the `{dev,train}` set. The `label_acc` files store a single float: the part-of-speech tagging accuracies on the `{dev,train}` set. The `params` file stores the probe parameters.
    
1. Make a minimal change to the config file, say replacing the `roberta-base` model with another model, specified by its huggingface identifier string.
  
## YAML-centric Design

This codebase revolves around the `yaml` configuration files that specify experiment settings.
Intended to minimize the amount of experiment logic code needed to swap out new `Probe`, `Loss`, or `Model` classes when extending the repository, all python classes defined in the codebase are actually constructed with the `yaml` loading process.

This is mostly documented in the `pyyaml` docs, [here](https://pyyaml.org/wiki/PyYAMLDocumentation), but briefly, consider the following config snippet:

```
cache: &id_cache !WholeDatasetCache
  train_path: &idtrainpath example/data/en_ewt-ud-sample/en_ewt-ud-train.conllu 
  dev_path: &iddevpath example/data/en_ewt-ud-sample/en_ewt-ud-dev.conllu
  test_path: &idtestpath example/data/en_ewt-ud-sample/en_ewt-ud-test.conllu 
```

When the `yaml` config is loaded, it result in a dictionary with the key `cache`.
The fun magic part is that `!WholeDatasetCache` references the code in `cache.py`, wherein the `WholeDatasetCache` class has the class attribute `yaml_tag = !WholeDatasetCache`. The `train_path`, `dev_path`, `test_path` are the arguments to this class's `__init__` function. Because of this, the value stored at key `cache` is an instance of `WholeDatasetCache`, constructed during `yaml` loading with the arguments provided.

All experiment objects -- `Probe`s, `Model`s, `Dataset`s, are constructed during `yaml` initialization in the same way.
Because of this, the logic for running an experiment -- in `experiment.py` -- is short.

### Some yaml basics

If you're not familiar with `yaml`, it's worthwhile to take a peek at the documentation.
We make frequent use of the referencing feature of yaml -- the ability to give an object in the `.yaml` config file an identifier, and place the same object elsewhere in the config file by referencing the identifier.

Making the label looks like:
```
input_fields: &id_input_fields
  - id
  - form
```
where the ampersand in `&id_input_fields` indicates the registration of an identifier; this object can then be placed elsewher in the config through

```
fields: *id_input_fields
```
where the asterisk in `*id_input_fields` indicates the reference of the object.

### Limiting logic in `__init__` due to yaml use

While the `yaml` object construction design decision makes it transparent which objects will be used in the course of a given experiment (instead of if/else/case statements that grow with the codebase scope), it adds a somewhat annoying consideration when writing code for these classes.

Stated briefly, all you can do in the `__init__` functions of your classes is assign arguments as instance variables, like `self.thing = thing`; you cannot run any code that relies on `thing` being an already-constructed object.

In more depth, the `yaml` loading process doesn't provide a guarantee on what order objects will be constructed. But we refer to objects (like the `input_fields` list) in constructing other objects, through `yaml` object reference. (Since, say, the `dataset` classes need to know what the `input_fields` list is.) So, when going through `yaml` loading, we do call `__init__` functions (see `utils.py`), but we are just passing around references and doing simple computation that doesn't depend on other `yaml`-constructed objects.

This means, somewhat unfortunately, that setup-style functionality, like checking the validity of cache files, for the `dataset` classes, has to be run at some time other than `__init__`. In practice, we check a check-for-setup condition into the functions that need the setup to have been run.

This toolkit is intended to be easily extensible, and allow for quick swapping of experimental components.
As such, the code is split into an arguably reasonable class layout, wherein one can write a new `Probe` or new `Loss` class somewhat easily.
More unusually, 

## Code layout and config runthrough

In this section we walk through the example configuration file and explain the classes associated with each component.
Each of these subsections refers to an object constructed during `yaml` loading, which is a "top-level" object, available in the loaded yaml config.

### Input-fields
Input-fields, for `conll`-formtted files, provides string labels for the columns of the file.
```
input_fields: &id_input_fields
  - id
  - form
  - lemma
  - upos
  - ptb_pos
  - feats
  - dep_head
  - dep_rel
  - None
  - misc
```
These identifiers will be used to pull the data of a column in the `AnnotationDataset` class; we'll go over this when we get to the `dataset` part of the config.

### cache
The cache object does some simple filesystem timestamp checking, and non-foolproof lock checking, to determine whether cache files for each dataset should be read from, or written to.
This is crucial for running many experiments with Huggingface `transformers` models, since the tokenization and alignment of subword tokens to corpus tokens takes more time than running the experiment itself once loaded.

```
cache: &id_cache !WholeDatasetCache
  train_path: &idtrainpath scripts/ontonotes_scripts/train.ontonotes.withdep.conll
  dev_path: &iddevpath scripts/ontonotes_scripts/dev.ontonotes.withdep.conll
  test_path: &idtestpath scripts/ontonotes_scripts/test.ontonotes.withdep.conll
```

Note that we make reference ids for both the WholeDatasetCache object itself and for the `{train,dev,test}` file paths, so we can use these later.

### disk_reader
The `Reader` objects are written to handle the oddities of a given filetype. The `OntonotesReader` object, for example, reads `conll` files, turning lines into sentences (given the `input_fields` object, above), while the `SST2Reader` object knows how to read `label\TABtokenized_sentence` data, as given by the `SST2` task of the GLUE benchmark.

```
disk_reader: !OntonotesReader &id_disk_reader
  args:
    device: cpu
  train_path: *idtrainpath 
  dev_path: *iddevpath 
  test_path: *idtestpath 
```
The `args` bit here is sort of a vestigal part of earlier code design; its only member, the `device`, is used whenver PyTorch objects are involved, to put tensors on the right device. Note how it references the dataset filepaths that were registered in the cache part of the config.

### dataset
The `ListDataset` object is always the top-level object of the `dataset` key; its job is to gather together output labels, and all of the input types, concatenate together the input, and yield minibatches for training and evaluation.

```
dataset: !ListDataset
  args:
    device: cpu
  data_loader: *id_disk_reader
  output_dataset: !AnnotationDataset
    args:
      device: cpu
    task: !TokenClassificationTask
      args:
        device: cpu
      task_name: ptb_pos
      input_fields: *id_input_fields
  input_datasets:
    - !HuggingfaceData
      args:
        device: cpu
        #model_string: &model1string google/bert_uncased_L-2_H-128_A-2
      model_string: &model1string google/bert_uncased_L-4_H-128_A-2
      cache: *id_cache
  batch_size: 5 
```
It is given the `DataLoader` from above so it can read data from disk.
It has a single specified `Dataset` for its output, here an `AnnotationDataset`.
The `AnnotationDataset` given here takes in a `Task` object -- here a `TokenClassificationTask`, to provide the labels for the output task. the `TokenClassificationTask` provides a label, using the `task_name` to pick out a column from the conll input file, as labeled by the `input_fields` list.

The `input_datasets` argument is a list of `Dataset` objects. All of these datasets' representations are bundled together by the `ListDataset`. Here, we only have one element in the list, a `HuggingfaceData` object, which runs the huggingface model specified by the `model_string`, but we could add a representation by adding another entry to the list.
The `HuggingfaceData` tokens and subword-to-corpus token alignment matrices will be read or written according to the `cache` given.

The `Dataset` generates (subword) tokens and alignment matrices, or label indices -- whatever a model needs as input.

Note that tasks like part-of-speech and dependency label, which have independent token-level labels, are easily exchangable in the `TokenClassificationTask`. But to run a task like named entity recognition, with its specialized specification of entity-level annotation (and evaluation, later), specialized classes are needed, like `NERClassificationTask`.

### model

For each dataset in `input_datasets`, a corresponding model takes the raw tokens provided by a `Dataset`, and runs the corresponding model to turn the input into a representation.
So, a `HuggingfaceData` above corresponds to a `HuggingfaceModel` here.

```
model: !ListModel
  args: 
    device: cpu
  models:
    - !HuggingfaceModel
        args: 
          device: cpu
        model_string: *model1string
        trainable: False
        index: 1
```
The `HuggingfaceModel` class runs the transformer model, and provides the representations of the layer at index `index`.
The `trainable` flag specifies whether to backprogate gradients back through the model and update its weights during training.

### probe
The `Probe` classes turn the representations given by `Model` classes into the logits of a distribution over the labels of the output task.

```
probe: !OneWordLinearLabelProbe
  args:
    device: cpu
  model_dim: 128
  label_space_size: 50
```
Somewhat unfortunately, it needs to be explicitly told what input and output dimensionality to expect.

### regimen
The regimen specifies a training procedure, with learning rate decay, loss, etc. Most of this is hard-coded right now to sane defaults.

```
regimen: !ProbeRegimen
  args:
    device: cpu
  max_epochs: 50
  params_path: params
  reporting_root: &id_reporting_root example/pos-bert-base.yaml.results
  eval_dev_every: 10
```
There's only one trainer as of now.
By convention, I put results directories at `<path_to_config>.results`. The `params_path` is relative to `reporting_root`.

### reproter
The reporter class takes predictions at the end of training, and reports evaluation metrics.

```
reporter: !IndependentLabelReporter
  args:
    device: cpu
  reporting_root: *id_reporting_root
  reporting_methods:
    - label_accuracy
    - v_entropy
```
For each of the strings in `reporting_methods`, a reporter function (which is specified by a hard-coded map from reporting string to function) is run on the data. The result of the metric is written to `<reporting_root>/<split>.<reporting_string>`.

Note that some reporters and metrics are specialized to a task.
For example, SST2 has its own `SST2Reporter` (though it's really just a sentence-level classification reporter)
and named entity recognition has its own `NERReporter`, which calls the Stanza library's NER evaluation script.


## Config recipes

### Named Entity Recognition config recipe
For an example of an NER config (e.g., using span-based eval), see

        configs/round1/named_entities/roberta768/layer0.yaml

### SST2 config recipe
For an example of a sentiment config (e.g., averaging the word embeddings for a sentence embedding), see

        configs/round1/sst2/roberta768/layer0.yaml

# Data preparation

## Ontonotes

See the `scripts/ontonotes_scripts` directory for notes on how we prep ontonotes.
If you don't care, and just want the data, it's as easy as:

Let `ldc_ontonotes_path` be the path to your LDC download of `Ontonotes 5.0`, that is, `LDC2013T19`. Mine looks like `/scr/corpora/ldc/2013/LDC2013T19/ontonotes-release-5.0/data/files/data/`.

Then all you have to run is

```
cd scripts/ontonotes_scripts
ldc_ontonotes_path=/scr/corpora/ldc/2013/LDC2013T19/ontonotes-release-5.0/data/files/data/
bash prep_ontonotes.sh $ldc_onotonotes_path
```

Nice. 

Statistics:

|                     | Train     | Dev     | Test    | Conll-2012-test |
|---------------------|-----------|---------|---------|-----------------|
| Sentences           | 111,707   | 15,161  | 11,696  | 9,479           |
| Tokens              | 2,100,642 | 292,052 | 216,628 | 169,547         |
| Last annotated line | 2209566   | 306802  | 228053  | 179057          |
