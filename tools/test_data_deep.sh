#!/bin/sh

rm -f wine-test.ds

# creates training dataset for nntool

./dstool -create wine-test.ds
./dstool -create:13:input wine-test.ds
./dstool -create:1:output wine-test.ds
./dstool -list wine-test.ds
./dstool -import:0 wine-test.ds wine_measurements.in
./dstool -import:1 wine-test.ds wine_measurements.out
# ./dstool -padd:0:meanvar wine-test.ds
# ./dstool -padd:0:pca wine-test.ds
# ./dstool -padd:1:meanvar wine-test.ds

./dstool -list wine-test.ds

# uses nntool trying to learn from dataset

# ARCH=13-1000-1000-1

ARCH=13-100-1

# ./nntool --deep=gaussian --samples 100 -v wine-test.ds $ARCH winenn.cfg grad
# ./nntool --purelinear --time 100 -v wine-test.ds $ARCH winenn.cfg grad
./nntool  --pseudolinear --time 100 -v wine-test.ds $ARCH winenn.cfg grad

# ./nntool -v --time 10 wine-test.ds 13-1 winenn.cfg random

# testing

./nntool -v wine-test.ds $ARCH winenn.cfg use

# predicting [stores results to dataset]

cp -f wine-test.ds wine-pred.ds
./dstool -clear:1 wine-pred.ds
# ./dstool -remove:1 wine-pred.ds

./nntool -v wine-pred.ds $ARCH winenn.cfg use

./dstool -list wine-test.ds
./dstool -list wine-pred.ds

./dstool -print:1 wine-pred.ds
tail wine_measurements.out

