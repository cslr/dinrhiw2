#!/bin/sh

rm -f numbers.ds

# creates training dataset for nntool

./dstool -create          numbers.ds
./dstool -create:2:input  numbers.ds
./dstool -create:1:output numbers.ds
./dstool -list            numbers.ds
./dstool -import:0        numbers.ds numbers.in
./dstool -import:1        numbers.ds numbers.out
./dstool -padd:0:meanvar  numbers.ds
# ./dstool -padd:0:pca iris-test.ds
./dstool -padd:1:meanvar  numbers.ds

./dstool -list numbers.ds

# uses nntool trying to learn from dataset

ARCH="2-10-10-1"

./nntool -v --samples 5000 --overfit numbers.ds $ARCH numbers-nn.cfg parallellbfgs

##################################################
# testing

./nntool -v numbers.ds $ARCH numbers-nn.cfg use

##################################################
# predicting [stores results to dataset]

cp -f numbers.ds numbers-pred.ds
./dstool -clear:1 numbers-pred.ds
# ./dstool -remove:1 wine-pred.ds

./nntool -v numbers-pred.ds $ARCH numbers-nn.cfg use

./dstool -list numbers.ds
./dstool -list numbers-pred.ds

./dstool -print:1 numbers-pred.ds
tail numbers.out


