#!/bin/sh
#
# TITANIC training dataset
#

DSTOOL="./dstool"
NNTOOL="./nntool"
DSFILE="titanic-train.ds"
DSFILE_PRED="titanic-test.ds"
DATAINPUT="titanic_train_input.out"
DATAOUTPUT="titanic_train_output.out"
DATAPRED="titanic_test_input.out"
DATAPREDOUT="titanic_test_output.out"
NNFILE="titanic-nn.cfg"
CSVPRED="titanic_pred.csv"


# create empty datafile
$DSTOOL -create $DSFILE
$DSTOOL -create:10:input $DSFILE
$DSTOOL -create:1:output $DSFILE

#populate it with data preprocess using mean=0,var=1 mean/variance removal
$DSTOOL -import:0 $DSFILE $DATAINPUT
$DSTOOL -import:1 $DSFILE $DATAOUTPUT
$DSTOOL -padd:0:meanvar $DSFILE
$DSTOOL -padd:1:meanvar $DSFILE

#######################################################################
# use nntool attempting to learn the dataset using L-BFGS/GRAD learning
ARCH="10-50-1"

$NNTOOL -v --samples 1000 $DSFILE $ARCH $NNFILE grad

#######################################################################
# test NN performance

$NNTOOL -v $DSFILE $ARCH $NNFILE use

#######################################################################
# predict scores

cp -f $DSFILE $DSFILE_PRED
$DSTOOL -clear:0 $DSFILE_PRED
$DSTOOL -clear:1 $DSFILE_PRED

# import new data to DSFILE_PRED
$DSTOOL -import:0 $DSFILE_PRED $DATAPRED

# calculates predictions
$NNTOOL -v $DSFILE_PRED $ARCH $NNFILE use

$DSTOOL -export:1 $DSFILE_PRED $DATAPREDOUT

#######################################################################
# converts $DATAPREDOUT TO KAGGLE SUBMISSION CSVFILE

./titanic_kaggle_export.py

