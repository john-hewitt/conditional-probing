#!/bin/bash
dir=$1

if [ -f 'corenlp.env' ]; then
    source corenlp.env
else
    echo "Please run setup_corenlp.sh first!"
    exit
fi

for file in $(find $dir -name "*gold_parse"); do
    echo Converting $file ...
    java -mx8g -cp "${CORENLP_HOME}" edu.stanford.nlp.trees.ud.UniversalDependenciesConverter -treeFile ${file} > ${file}.dep.conllu
done
