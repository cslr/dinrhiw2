#!/bin/sh

# creates training dataset for nntool
make gendata

NNTOOL="./nntool"
DSTOOL="./dstool"
GENDATA="./gendata"

$GENDATA 5

$DSTOOL -create gendata-test.ds
$DSTOOL -create:5:input gendata-test.ds
$DSTOOL -create:1:output gendata-test.ds
$DSTOOL -list gendata-test.ds
$DSTOOL -import:0 gendata-test.ds gendata_scoring.csv
$DSTOOL -import:1 gendata-test.ds gendata_scoring_correct.csv
$DSTOOL -padd:0:meanvar gendata-test.ds
# $DSTOOL -padd:0:pca wine-test.ds
$DSTOOL -padd:1:meanvar gendata-test.ds

$DSTOOL -list gendata-test.ds

# uses nntool trying to learn from dataset

ARCH="5-1000-1"

#$NNTOOL -v wine-test.ds $ARCH winenn.cfg mix
# $NNTOOL -v wine-test.ds $ARCH winenn.cfg lbfgs

################## $NNTOOL -v --samples 2000 wine-test.ds $ARCH winenn.cfg grad

$NNTOOL -v --time 600 gendata-test.ds $ARCH gendatann.cfg grad

# $NNTOOL -v --time 10 wine-test.ds $ARCH winenn.cfg random

# testing

$NNTOOL -v gendata-test.ds $ARCH gendatann.cfg use

# predicting [stores results to dataset]

cp -f gendata-test.ds gendata-pred.ds
$DSTOOL -clear:1 gendata-pred.ds
# $DSTOOL -remove:1 wine-pred.ds

$NNTOOL -v gendata-pred.ds 13-13-1 gendatann.cfg use

#$DSTOOL -list wine-test.ds
#$DSTOOL -list wine-pred.ds

#$DSTOOL -print:1 gendata-pred.ds
#tail gendata_scoring_correct.csv
