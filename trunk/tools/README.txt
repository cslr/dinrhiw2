
The primary programs created using "make" command
(after make all install in the root directory one
level higher to install dinrhiw library) are:
nntool and dstool. (+ make install to instal dinrhiw-tools).

After this you then preprocess data text files using perl scripts:

process_data2.pl
process_data3.pl

Which create proper text files with data that can be imported into
datasets files using dstool. These datasets files are then read by
nntool (neural network code) that can be used for machine learning
relationships from the data.

The test scripts to test neural network training code are:

test_data.sh
test_data_parallel.sh
test_data2.sh
test_data2_parallel.sh
test_data2_parallel_random.sh
test_data3.sh
test_data3_parallel.sh

which use both dstool and nntool and print some results.

The scripts do PCA preprocessing and mean variance removal
preprocessing which means that data is something like normally
distributed data with zero mean = Normal(0, I) when it is fed
to the neural network code.

This means that mean error of the training process is often
usable to predict learning results. Values below 0.01 mean
that neural network errors are close to minimum and results
are usable and mean error rates higher than it mean that
the neural network do NOT converge (dataset 3) and it cannot
be used to predict future outcomes.

The "best" gradient descent code in nntool is "lbfgs" which
starts multistart parallel L-BFGS searches with NUMCORES threads
where NUMCORES is number of cores or hyperthreading units in CPU.

It keeps doing Limited memory BFGS optimization with early stopping
again and again from semirandomly chosen starting points until
timeout or number of iterations has been reached.

DATA
----

Machine learning datasets are from UCI Machine learning repository:

http://archive.ics.uci.edu/ml/

