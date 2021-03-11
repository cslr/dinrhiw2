#!/bin/sh

# creates training dataset for nntool

NNTOOL="nntool"
DSTOOL="dstool"

$DSTOOL -list simpleproblem.ds

# uses nntool trying to learn from dataset

## ARCH="4-4"
# ARCH="4-10-4"
ARCH="4-10-10-4"

################## $NNTOOL -v --samples 2000 wine-test.ds $ARCH winenn.cfg grad

$NNTOOL -v --noresidual --time 100 simpleproblem.ds $ARCH simpleproblem.cfg pgrad

# testing

$NNTOOL -v --noresidual simpleproblem.ds $ARCH simpleproblem.cfg use



