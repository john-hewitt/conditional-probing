#!/bin/bash

# Download the OntoNotes CoNLL2012 v4 split files,
# get things organized and remove unnecessary files.

echo "Downloading v4 split data..."
wget http://conll.cemantix.org/2012/download/conll-2012-train.v4.tar.gz
wget http://conll.cemantix.org/2012/download/conll-2012-development.v4.tar.gz
wget http://conll.cemantix.org/2012/download/test/conll-2012-test-key.tar.gz

# decompress things into conll-2012 folder
tar zxf conll-2012-train.v4.tar.gz
tar zxf conll-2012-development.v4.tar.gz
tar zxf conll-2012-test-key.tar.gz

# remove Arabic and Chinese, as well as unused files
rm -r conll-2012/*/data/*/data/chinese
rm -r conll-2012/*/data/*/data/arabic
rm -r conll-2012/*/data/*/data/english/annotations/*/*/*/*auto_*

# remove the pt folders, which do not contain NER annotations
rm -r conll-2012/*/data/*/data/english/annotations/pt

# move things into a data folder in the root
mv conll-2012/v4/data conll-2012/
rm -r conll-2012/v4

echo "Split data processed."
