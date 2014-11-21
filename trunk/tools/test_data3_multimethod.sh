
#!/bin/sh

rm -f commviol-test.ds

# creates training dataset for nntool

./dstool -create commviol-test.ds
./dstool -create:141:input commviol-test.ds
./dstool -create:4:output commviol-test.ds
./dstool -list commviol-test.ds
./dstool -import:0 commviol-test.ds commviol.in
./dstool -import:1 commviol-test.ds commviol.out
./dstool -padd:0:meanvar commviol-test.ds
./dstool -padd:0:pca commviol-test.ds
./dstool -padd:1:meanvar commviol-test.ds

./dstool -list commviol-test.ds

# uses nntool trying to learn from dataset
# (multistart gradient descent and then one large gradient descent reduction)

./nntool --samples 1000 --time 3600 -v commviol-test.ds 141-10-4 commviol-nn.cfg parallelgrad
./nntool --samples 10000 --load -v commviol-test.ds 141-10-4 commviol-nn.cfg grad
./nntool --samples 2000 --load -v commviol-test.ds 141-10-4 commviol-nn.cfg bayes

##################################################
# testing

./nntool -v commviol-test.ds 141-10-4 commviol-nn.cfg use

##################################################
# predicting [stores results to dataset]

cp -f commviol-test.ds commviol-pred.ds
./dstool -clear:1 commviol-pred.ds
# ./dstool -remove:1 wine-pred.ds

./nntool -v commviol-pred.ds 141-10-4 commviol-nn.cfg use

./dstool -list commviol-test.ds
./dstool -list commviol-pred.ds

./dstool -print:1 commviol-pred.ds
tail commviol.out

echo "ERRORS ARE HIGH HERE AND WE DO *NOT* CONVERGE"


