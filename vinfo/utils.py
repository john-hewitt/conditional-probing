"""
Utilities for determining file locations or names
from the configuration specification
"""
from yaml import YAMLObject

IGNORE_LABEL_INDEX = -100
TRAIN_STR = 'train'
DEV_STR = 'dev'
TEST_STR = 'test'

PTB_UNIVERSAL_CONVERSION_STRING = 'ptb_to_upos'
WSD_COARSENING_CONVERSION_STRING = 'wsd_coarse'

def get_results_root(config):
  pass

def get_experiment_dir(config):
  pass

def get_default_ontonotes_fieldnames():
  """
  """

class InitYAMLObject(YAMLObject):

  @classmethod
  def from_yaml(cls, loader, node):
    """
    Convert a representation node to a Python object.
    """
    arg_dict = loader.construct_mapping(node, deep=True)
    print('Constructing', cls)
    return cls(**arg_dict)

# The map as given here is from
# https://raw.githubusercontent.com/slavpetrov/universal-pos-tags/master/en-ptb.map
ptb_to_univ_map = {'!': '.', '#': '.', '$': '.', "''": '.', '(': '.', ')': '.',
                   ',': '.', '-LRB-': '.', '-RRB-': '.', '.': '.', ':': '.',
                   '?': '.', 'CC': 'CONJ', 'CD': 'NUM', 'CD|RB': 'X', 'DT': 'DET',
                   'EX': 'DET', 'FW': 'X', 'IN': 'ADP', 'IN|RP': 'ADP', 'JJ': 'ADJ',
                   'JJR': 'ADJ', 'JJRJR': 'ADJ', 'JJS': 'ADJ', 'JJ|RB': 'ADJ', 
                   'JJ|VBG': 'ADJ', 'LS': 'X', 'MD': 'VERB', 'NN': 'NOUN', 'NNP': 'NOUN',
                   'NNPS': 'NOUN', 'NNS': 'NOUN', 'NN|NNS': 'NOUN', 'NN|SYM': 'NOUN', 
                   'NN|VBG': 'NOUN', 'NP': 'NOUN', 'PDT': 'DET', 'POS': 'PRT',
                   'PRP': 'PRON', 'PRP$': 'PRON', 'PRP|VBP': 'PRON', 'PRT': 'PRT',
                   'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV', 'RB|RP': 'ADV', 'RB|VBG': 'ADV',
                   'RN': 'X', 'RP': 'PRT', 'SYM': 'X', 'TO': 'PRT', 'UH': 'X',
                   'VB': 'VERB', 'VBD': 'VERB', 'VBD|VBN': 'VERB', 'VBG': 'VERB',
                   'VBG|NN': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBP|TO': 'VERB',
                   'VBZ': 'VERB', 'VP': 'VERB', 'WDT': 'DET', 'WH': 'X',
                   'WP': 'PRON', 'WP$': 'PRON', 'WRB': 'ADV', '``': '.'}
# But it doesn't include all the tags in Ontonotes... so we make decisions about them here.
ptb_to_univ_map['HYPH'] = '.' # Suggested by the behavior of the EWT treebank
ptb_to_univ_map['AFX'] = 'X' # Suggested by the behavior of the EWT treebank
ptb_to_univ_map['XX'] = 'X' # Speech influencies

coarse_wsd_map = {# Tags that are mapped to the ignore tag (counts <10 in train set)
                  "11.26":"-", "11.4":"-", "12.12":"-", "12.7":"-", "13.8":"-",
                  "16.10":"-", "16.8":"-", "17.9":"-", "31":"-", "32":"-", "33":"-",
                  "5.7":"-", "6.2":"-", "6.4":"-", "7.14":"-", "7.19":"-",
                  "7.24":"-", "7.27":"-", "7.6":"-", "11.10":"-", "11.17":"-",
                  "11.8":"-", "11.9":"-", "13.17":"-", "16.11":"-", "16.9":"-",
                  "25":"-", "5.11":"-", "5.9":"-", "7.12":"-", "7.16":"-",
                  "7.17":"-", "7.26":"-", "7.31":"-", "7.5":"-", "11.15":"-",
                  "11.19":"-", "11.24":"-", "11.31":"-", "12.13":"-", "13.1":"-",
                  "13.21":"-", "13.6":"-", "17.6":"-", "29":"-", "5.12":"-",
                  "7.32":"-", "7.7":"-", "18":"-", "23":"-", "6.1":"-",
                  "7.13":"-", "7.21":"-", "7.29":"-", "12.6":"-", "13.2":"-",
                  "24":"-", "5.2":"-", "5.4":"-", "7.11":"-", "7.8":"-",
                  "8.12":"-", "11.12":"-", "11.32":"-", "16.2":"-", "16.3":"-",
                  "5.5":"-", "7.9":"-", "11.38":"-", "11.7":"-", "12.8":"-",
                  "11.20":"-", "11.33":"-", "17":"-", "11.13":"-", "11.23":"-",
                  "11.6":"-", "13.9":"-",

                  # In dev or test (didn't check which) but not in train
                  "12.1":"-",
                  "17.1":"-",
                  "5.1":"-",
                  "7.10":"-",
                  "7.18":"-",

                  # Tags that are kept as-is (counts >=10 in train set)
                  "11.5":"11.5", "26":"26", "11.1":"11.1", "16.1":"16.1", "16.5":"16.5",
                  "7.15":"7.15", "7.28":"7.28", "13.5":"13.5", "7.3":"7.3", "20":"20",
                  "21":"21", "19":"19", "11.3":"11.3", "16.4":"16.4", "5.8":"5.8",
                  "5.6":"5.6", "13.4":"13.4", "11.37":"11.37", "7.4":"7.4", "7.1":"7.1",
                  "7.2":"7.2", "16":"16", "11.2":"11.2", "14.1":"14.1", "13":"13",
                  "15":"15", "14":"14", "11":"11", "10":"10", "9":"9",
                  "8":"8", "12":"12", "7":"7", "6":"6", "5":"5",
                  "4":"4", "3":"3", "2":"2", "1":"1", "-":"-"
                 }


def get_conversion_dict(conversion_name):
  """Retrieves a hard-coded label conversion dictionary.

  When coarsening the label set of a task based on a predefined
  conversion scheme like Penn Treebank tags to Universal PoS tags,
  this function provides the map, out of a fixed list of known
  maps addressed by a keyword string.
  """
  if conversion_name == PTB_UNIVERSAL_CONVERSION_STRING:
    return ptb_to_univ_map
  elif conversion_name == WSD_COARSENING_CONVERSION_STRING:
    return coarse_wsd_map
  else:
    raise ValueError("Unknown conversion name: {}".format(conversion_name))
