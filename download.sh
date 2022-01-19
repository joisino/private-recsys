#!/bin/bash

git submodule update --init --recursive
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Home_and_Kitchen_5.json.gz
gunzip reviews_Home_and_Kitchen_5.json.gz
python preprocessing.py home
python preprocessing.py adult
wget --no-check-certificate https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
mkdir hetrec
unzip hetrec2011-lastfm-2k.zip -d hetrec
python preprocessing.py hetrec

wget --no-check-certificate https://files.grouplens.org/datasets/movielens/ml-100k.zip 
wget --no-check-certificate https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-100k.zip
unzip ml-1m.zip
