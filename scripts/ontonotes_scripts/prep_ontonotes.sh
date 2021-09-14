#!/bin/bash
# Preps data from ontonotes as used in _Conditional Variational Information Probes_
# Attempts to describe the entire data processing process, as there seems to be no
# single accepted way to process ontonotes.
#
# As such, this dataset is likely not exactly comparable to any prior work.
# It's unclear what prior work is comparable anyway, due to subtle differences in the 
# splits, preprocessing, excluded data etc.
# We're okay with this. If you run this script, your work can be comparable to this work.

## Input validation
# Something if args < 2 then fail and print usage

##### Step 0
## retrieve the raw LDC release of Ontonotes 5.0, that is, LDC2013T19 
## https://catalog.ldc.upenn.edu/LDC2013T19

# For example, ldc_ontonotes=/scr/corpora/ldc/2013/LDC2013T19/ontonotes-release-5.0/data/files/data/
ldc_ontonotes=$1 # Input as argument to this script
#ldc_ontonotes=/scr/corpora/ldc/2013/LDC2013T19/ontonotes-release-5.0/data/files/data/

##### Step 1
## Convert the raw LDC release to conll format, with PoS, NER, constituents, SRL, coref

# First, download the "skeleton" conll files that do not include the actual words 
# (due to (silly?) copyright problems)
# but, crucially, do provide the train/dev/test splits
wget https://github.com/ontonotes/conll-formatted-ontonotes-5.0/archive/v12.tar.gz
tar xzvf v12.tar.gz

# Second, merge the raw words from the Ontonotes release with the skeleton files to get _conll files
bash skeleton2conll.sh -D ${ldc_ontonotes} conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/

##### Step 2
## To get dependency parses for Ontonotes, we convert the given constituency trees to dependency trees using
## Stanford CoreNLP (v3.3 for Stanford Dependencies and v4.0 for Universal Dependencies)
## but to add complexity, the conll-formatted version of the dataset actually removes 
## all constituencies with the EDITED tag, so if you just convert the original constituency trees, the two
## datasets don't match. Instead, we convert the conll "parse bits" back to constituency tree files,

# The following script produces a constituency parse tree file for each conll file (from the "parse bit")
bash conll2parse.sh conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/

##### Step 3
## Next we convert the recovered constituency parse trees (that do not include EDITED-tagged constituents)
## to dependency trees

# First we set up Stanford CoreNLP from scratch... will take a second
bash setup_corenlp.sh 4.0

# Next, we use CoreNLP to generate dependency parse tree files (in conllx format)
# for each of the constituency parse tree files
bash convert_splits_to_depparse.sh conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/

##### Step 4
## Next, we align the conll files with (PoS, NER, constituents, SRL, Coref) that is, the ones
## from the original skeleton files, (orig_conll)
## with the conllx files with (PoS, Deprel, head) labels (dep_conll).

# This process is a bit convoluted and involves the following components:
# Remove comments from the orig_conll files
# <grep thing>
# Replace spaces in orig_conll files with tabs
# <sed thing>
# Remove single-word sentences where the word is XX, from orig_conll,
# since the dependency construction just breaks and omits these (useless anyway) sentences.
# <complex thing>
# Paste them together
# <paste thing>
# Remove tabs from otherwise-empty lines
# <sed thing>
paste <( cat conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/development/data/english/annotations/*/*/*/*.dep.conllu ) <( cat conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/development/data/english/annotations/*/*/*/*.gold_conll  | grep -P -v "^#" | sed "s/  */\t/g"  | grep -P -v ".*0\t0\tXX"  | cat -s ) | sed "s/^\t$//" > dev.ontonotes.withdep.conll
paste <( cat conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/*/*/*/*.dep.conllu ) <( cat conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/*/*/*/*.gold_conll  | grep -P -v "^#" | sed "s/  */\t/g"  | grep -P -v ".*0\t0\tXX"  | cat -s ) | sed "s/^\t$//" > train.ontonotes.withdep.conll
paste <( cat conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/test/data/english/annotations/*/*/*/*.dep.conllu ) <( cat conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/test/data/english/annotations/*/*/*/*.gold_conll  | grep -P -v "^#" | sed "s/  */\t/g"  | grep -P -v ".*0\t0\tXX"  | cat -s ) | sed "s/^\t$//" > test.ontonotes.withdep.conll
paste <( cat conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/conll-2012-test/data/english/annotations/*/*/*/*.dep.conllu ) <( cat conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/conll-2012-test/data/english/annotations/*/*/*/*.gold_conll  | grep -P -v "^#" | sed "s/  */\t/g"  | grep -P -v ".*0\t0\tXX"  | cat -s ) | sed "s/^\t$//" > conll2012test.ontonotes.withdep.conll


#
###### Step (5)
## You're done! Wow that took too much work. Good thing it was automated.
