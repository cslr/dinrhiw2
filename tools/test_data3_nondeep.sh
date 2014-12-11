#!/bin/sh

rm -f commviol-test.ds

# creates training dataset for nntool

./dstool -create commviol-test.ds
./dstool -create:141:input commviol-test.ds
./dstool -create:4:output commviol-test.ds
./dstool -import:0 commviol-test.ds commviol.in
./dstool -import:1 commviol-test.ds commviol.out
./dstool -padd:0:meanvar commviol-test.ds
./dstool -padd:1:meanvar commviol-test.ds

# uses nntool trying to learn from dataset

./nntool -v --samples 1000 --overfit commviol-test.ds 141-20-4 commviol-nn.cfg lbfgs
# ./nntool -v --samples 1000 --overfit --negfb commviol-test.ds 141-20-4 commviol-nn.cfg lbfgs
# ./nntool -v --samples 1000 --negfb commviol-test.ds 141-20-4 commviol-nn.cfg parallellbfgs
#./nntool -v --samples 100 --load  --overfit commviol-test.ds 141-141-4-4 commviol-nn.cfg lbfgs
# ./nntool -v --samples 1000 --load commviol-test.ds 141-141-4-4 commviol-nn.cfg bayes

##################################################
# testing

./nntool -v commviol-test.ds 141-20-4 commviol-nn.cfg use

##################################################
# predicting [stores results to dataset]

cp -f commviol-test.ds commviol-pred.ds
./dstool -clear:1 commviol-pred.ds
# ./dstool -remove:1 wine-pred.ds

./nntool -v commviol-pred.ds 141-20-4 commviol-nn.cfg use

./dstool -print:1:2205:2214 commviol-pred.ds
tail commviol.out

echo "ERRORS ARE HIGH HERE AND WE DO *NOT* CONVERGE"
