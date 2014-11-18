/*
 * basic algoritms library
 * 
 *   basic datatypes
 *   class/function behaviour morphers
 *   tree structures and sorting
 *   misc utilities
 *   fast linear algebra library (PCA, SVD)
 *   data interpolation
 *   heuristical optimizers: GA, PSO
 *   basic cryptography
 *   feedforward neural networks
 *   clustering: SOM2D, KMeans
 *   
 * in progress/partially working
 *   simplex optimization
 *   crude probability distrubtion estimator
 *   timed_boolean
 */


#ifndef lib_dinrhiw_h
#define lib_dinrhiw_h

#include "BFGS.h"

#include "array.h"
#include "static_array.h"
#include "dynamic_array.h"
#include "search_array.h"
#include "container.h"
#include "compressable.h"
#include "printable.h"
#include "data_source.h"
#include "chainhash.h"
#include "hash_table.h"
#include "stack.h"
#include "dummystack.h"
#include "queue.h"
#include "priority_queue.h"
#include "dynamic_bitset.h"
#include "point.h"
#include "ownexception.h"

#include "function_access_control.h"
#include "dynamic_ctor_protection.h"
#include "singleton.h"
#include "singleton_list.h"

#include "augmented_data.h"
#include "tree.h"
#include "avltree.h"
#include "avlnode.h"
#include "rbtree.h"
#include "rbnode.h"
#include "binary_tree.h"
#include "btree_node.h"

#include "fast_radix.h"
#include "fast_gradix.h"
#include "fast_radixv.h"
#include "radix_sort.h"
#include "radix_sortv.h"
#include "insertion.h"

#include "ETA.h"
#include "linear_ETA.h"
#include "MemoryCompressor.h"
#include "conffile.h"
#include "unique_id.h"

#include "atlas.h"
#include "number.h"
#include "blade_math.h"
#include "atlas_primitives.h"
#include "integer.h"
#include "real.h"
#include "modular.h"
#include "gcd.h"
#include "primality_test.h"
#include "matrix.h"
#include "vertex.h"
#include "quaternion.h"
#include "gmatrix.h"
#include "gvertex.h"
#include "norms.h"
#include "linear_algebra.h"
#include "linear_equations.h"
#include "matrix_rotations.h"
#include "eig.h"
#include "conversion.h"
#include "correlation.h"
/// #include "BFGS.h"

#include "bezier.h"
#include "bezier_density.h"
#include "bezier_surface.h"
#include "hermite.h"

#include "PSO.h"
#include "GeneticAlgorithm.h"
#include "GeneticAlgorithm2.h"

#include "Cryptosystem.h"
#include "DataConversion.h"
#include "AES.h"
#include "DES.h"
#include "DSA.h"
#include "PAD.h"
#include "RSA.h"
#include "SHA.h"

#include "dataset.h"
#include "activation_function.h"
#include "backpropagation.h"
#include "neuralnetwork.h"
#include "neuron.h"
#include "neuronlayer.h"
#include "nnPSO.h"
#include "dnnPSO.h"
#include "nn_iterative_correction.h"

#include "nnetwork.h"
#include "NNGradDescent.h"
#include "NNRandomSearch.h"

#include "KMeans.h"
#include "SOM2D.h"

#include "function.h"
#include "optimized_function.h"
#include "optimized_nnetwork_function.h"
#include "negative_function.h"
#include "stretched_function.h"
#include "identity_activation.h"
#include "odd_sigmoid.h"
#include "threshold.h"
#include "treshold.h"
#include "multidimensional_gaussian.h"
#include "gaussian.h"

#include "HMC.h"
#include "bayesian_nnetwork.h"
#include "GA3.h"

// in development [experimental]
// these do not really work well enough (yet)
#include "rbf.h"  
#include "hypervolume.h"
#include "pdftree.h"
#include "simplex.h"


#include "timed_boolean.h"
#include "superresolution.h"
#include "memory_access_timing.h"

// misc [used by internal testing]
#include "list_source.h"    // simple data_source
#include "test_function.h"
#include "test_function2.h"


//////////////////////////////////////////////////


  


#endif
