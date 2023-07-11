#!/usr/bin/env bash

# cd data/sent140 || exit

# if [ ! -f 'glove.6B.300d.txt' ]; then
#     wget http://nlp.stanford.edu/data/glove.6B.zip
#     unzip glove.6B.zip
#     rm glove.6B.50d.txt glove.6B.100d.txt glove.6B.200d.txt glove.6B.zip
#     mv glove.6B.300d.txt ../../enea_fl/models/glove.6B.300d.txt
# fi

cd enea_fl/models || exit

if [ ! -f 'glove.6B.50d.txt' ]; then
   wget http://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip
   rm glove.6B.300d.txt glove.6B.100d.txt glove.6B.200d.txt glove.6B.zip
fi

if [ ! -f embs.json ]; then
  python3 get_embs.py
fi