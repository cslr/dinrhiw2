<TeXmacs|1.0.1.21>

<style|generic>

<\body>
  <strong|Feedforward Neural Network Implementation Notes/Documentation>

  \;

  New neural network code is (will be) implemented with BLAS, level 1, 2, 3
  operations in C++.

  Properties/Reasons for new implementation:

  - Effieciency. The previous implementation was conceptially simple and
  object oriented but too slow.

  - By using ATLAS C BLAS routines for linear algebra performance should
  reach much closer toward theoretical peak performance. Except for non the
  linear activation function, almost everything can be wrote down as
  matrix*vertex or vertex operations. Implementation will be very efficient
  on x86 when MMX/SIMD/3DNow! operations are used for vector math.

  - need to implement good learning algorithm, instead of primitive basic
  backpropagation algorithm.

  - need to integrate neural network code with new whiteice::vertex and
  whiteice::matrix classes which use ATLAS CBLAS

  \;

  <strong|Implementation>

  \;

  <em|Neural Network / Neural Network Layer>

  Class whiteice::neuralnetwork is mostly ordered list of Neural Network
  Layers.

  Class whiteice::neuronlayer is implementation of single layer of network.

  Layer of networks can be written as

  <with|mode|math|y=\<varphi\>(W*x+b)> \ \ ,
  <with|mode|math|y\<in\>\<bbb-R\><rsup|N>,
  x\<in\>\<bbb-R\><rsup|M>,b\<in\>\<bbb-R\><rsup|N>,
  W\<in\>\<bbb-R\><rsup|N\<times\>M>>

  And <with|mode|math|\<varphi\>: \ \<bbb-R\><rsup|N>\<rightarrow\>\<bbb-R\><rsup|N>>

  \;

  <em|Backpropagation Learning>

  Backpropagation implementation is in class whiteice::backpropagation.

  Backpropagation algorithm can be formulated in linear algebra form. Update
  rule of the last layer can be written as

  <with|mode|math|W<rsub|n+1>=W<rsub|n>+\<eta\>*\<delta\>(n)*x(n)<rsup|T>> ,
  which also holds for any layer where <with|mode|math|x(n)> is input vector
  for the neuron layer

  <with|mode|math|b<rsub|n+1>=b<rsub|n>+\<eta\>*\<delta\>(n)>

  With

  local gradient vector <with|mode|math|\<delta\>(n)<rsup|T>=e(n)<rsup|T>\<odot\>\<nabla\><rsub|s>\<varphi\>(s)=[e(1,n)*\<varphi\><rsup|<rprime|'>>(s<rsub|1>)
  , \ e(2,n)*\<varphi\><rprime|'>(s<rsub|2>), \<ldots\>. ,
  e(N,n)*\<varphi\><rprime|'>(s<rsub|N>)]>.

  <with|mode|math|e(n)<rsup|T>=[e(1,n) , e(2,n), \<ldots\>. , e(N,n)]> and

  <with|mode|math|\<varphi\>(s)=[\<varphi\>(s<rsub|1>),
  \<varphi\>(s<rsub|2>), \<ldots\>\<ldots\>, \<varphi\>(s<rsub|N>)]>.

  \;

  Hidden Layers

  Propagation of local gradient can be calculated with

  <with|mode|math|\<delta\><rsub|L-1>(n)=[W<rsub|L,n><rsup|T>*\<delta\><rsub|L>(n)]\<odot\>\<nabla\><rsub|s>\<varphi\>(s<rsub|L-1>)>.

  \;

  So general BP algorithm in which is written on vector form\ 

  <with|mode|math|\<delta\><rsub|L-1>(n)=[W<rsub|L><rsup|T>(n)*\<delta\><rsub|L>(n)]\<odot\>\<nabla\><rsub|s>\<varphi\>(s<rsub|L>)>

  <with|mode|math|W<rsub|L>(n+1)=W<rsub|L>(n)+\<eta\>*\<delta\><rsub|L>(n)*x<rsub|L>(n)<rsup|T>>

  <with|mode|math|b<rsub|L>(n+1)=b<rsub|L>(n)+\<eta\>*\<delta\><rsub|L>(n)>

  Because updating of gradients use old W and updating of weights use old
  gradient one must save old gradient for the update. Alternative approach
  would be to calculate and save outer product and use only it but it takes
  too much space.

  \;

  Pseudocode CBLAS implementation

  cblas_gemv(next_grad = W^t * prev_grad)

  next_grad *= (valuewise) grad_activation(state_l)

  cblas_geru(W = W + loca_grad * input_grad_^T)

  cblas_Xaxpy(bias += learning_rate*local_gradient )

  \;

  <em|Other Learning Algorithms>

  Algorithms to implement

  - <em|Conjugate Gradient Learning> - this is <with|mode|math|O(W)>
  approximative second order algorithm

  - <em|Levenberg-Marguardt Learning> - this is <with|mode|math|O(W<rsup|2>)>
  approximative 2nd order algorithm

  \ \ with a prior information put into constraint matrix <with|mode|math|Q>

  \;

  \;

  \;

  \;

  <em|Levenberg-Marguardt Learning>

  Levenberg-Marguardt Optimization can be used to minimize the network error
  and is used by Matlab implementation for example. Algorithm is well studied
  and works fast by approximating gradient.

  IN THE FUTURE / Not done yet

  Implementation is based on paper Levenberg-Marquardt Learning and
  Regularization. Lai-Wan CHAN.

  LM-learning is simply <with|mode|math|\<Delta\>w=(J<rsup|T>J+\<gamma\>Q<rsup|T>Q)<rsup|-1>J<rsup|T>e>
  and is based on approximating function based on taylor expansions and then
  going toward approximated location of zero.

  <with|mode|math|J> is the jacobian matrix of NN parameters - exact form
  this is:

  OK - DO SOME MATH / THINKING what is the order of parameters and what is
  smart way to formulate problem so that <with|mode|math|J> matrix is easy
  etc. calculate.

  \;

  \;

  Regularized solution is \ depends on the choice of <with|mode|math|Q>
  matrix which is one of the parameters for the learning algorithm. When
  metalevel learning based on information sharing between simple neural
  networks will (hopefully) be implemented (need some theoretical and testing
  work), this <with|mode|math|Q> matrix must be one of the parameters (which
  also includes the choice of the activation function which leads to small
  set of parameters in neural network).

  \ - note to self: try to use these at first with 'best-matching strategy'
  in the history based on approximated information theoretic similarity
  between learnt neural network parameters, try to figure out what would be
  good way to 'sum' multiple past learnt neural networks which have similar
  information theoretic similarity/correlation, information theoretic
  similarity of two neural network (or any learning system) parameters
  <with|mode|math|I(W<rsub|1>,W<rsub|2>)> must be learnt to approximately
  from the distribution examples: features for distribution: this need work,
  \ must have\ 

  <with|mode|math|A)> distribution based <em|dimensioless> features

  <with|mode|math|B)> distribution based features of which dimension is based
  on the dimensions of examples. This must be done after all examples spaces
  are (cruely/forcefully) crunched into some predefined dimension
  <with|mode|math|D=1,2,3,4>(?) with PCA. This probably works somehow but
  some more clever approach would be better).

  Based on calculated features try to teach system which approximates
  <with|mode|math|I(W<rsub|1>,W<rsub|2>)> after training. So

  <with|mode|math|I<rprime|'>(W<rsub|1>,W<rsub|2>)=NN(features of
  (X<rsub|1>,Y<rsub|1>), features of (X<rsub|2>,Y<rsub|2>) )>.

  \;

  -- OLD

  'Best match strategy' simply finds out best match and if it's good enough,
  learnt <with|mode|math|W<rsub|old>> will be used as initial values (or mean
  of <with|mode|math|W<rsub|new problem>>) also used activation functions, Q
  etc. are used. Also some other alternatives etc. can be used.

  If nothing good enough is found then one needs to try out expensively many
  values of <with|mode|math|Q>, activation functions and pick for the best.
  In order to have information sharing here between learning systems yet
  another basic <with|mode|math|NN> (or RBF or SVM..) should be used which is
  teached to pick possible good paramters to try out.

  --- OLD

  Actually in order to merge past experience nicely it's probably easiest to
  teach ML with distribution features and good parameter values which are
  found by trying. Then this <with|mode|math|ML> can be used (if it learns
  anything...) to give out initial <with|mode|math|W,Q,activation function>
  which should work well for the given problem. Problem with
  <with|mode|math|W> and <with|mode|math|Q> are that they are depend on
  dimensions of data handled by <with|mode|math|ML>.\ 

  quick&dirty solution(?): implement ML (=machine learning) system with
  different number of outputs. )

  \;

  \;

  <strong|Preprocessing>

  Inputs should be decorrelated because it can be proven in certain special
  cases that without decorralation learning times grows
  <with|mode|math|O(2<rsup|N>)> and with uncorrelated inputs learning takes
  polynomial time (Find ref. to paper in 1995). Experience shows that this is
  good idea to do generally (Haykin).

  <em|Whitening of input>

  <with|mode|math|y=A*(x-m<rsub|x>)> \ , where
  <with|mode|math|A=B*\<Lambda\><rsub|R<rsub|x>><rsup|-1/2>X<rsup|H>> ,
  <with|mode|math|m<rsub|x>=E[x]>

  And <with|mode|math|R<rsub|x>=E[x*x<rsup|H>]=X*\<Lambda\><rsub|R<rsub|x>>*X<rsup|T>>
  \ , <with|mode|math|B=eye(\<sigma\><rsup|2>)>, where
  <with|mode|math|\<sigma\><rsup|2>> is a target variance vector.

  In this implementation it chosen <with|mode|math|\<forall\>n:
  \<sigma\><rsup|2>(n)=0.5<rsup|2>>, which puts data into a proper interval
  when the first layer of nodes has a sigmoid activation function. It's
  assumed that possible outliers have been removed before this preprocessing
  so that they cannot cause errors into statistics.

  In case of function approximation the final layer of a network should be
  linear (so range of output etc. aren't constrained) and for pattern
  classification it's good idea to have normal sigmoid activation function
  ([-1,1] range).

  \;

  <em|Calculation of eigenvectors and values of symmetric matrix>

  \;
</body>

<\initial>
  <\collection>
    <associate|paragraph width|6.5in>
    <associate|odd page margin|1in>
    <associate|page right margin|1in>
    <associate|page top margin|1in>
    <associate|reduction page right margin|0.7in>
    <associate|page type|letter>
    <associate|reduction page bottom margin|0.3in>
    <associate|even page margin|1in>
    <associate|reduction page left margin|0.7in>
    <associate|page bottom margin|1in>
    <associate|reduction page top margin|0.3in>
    <associate|language|finnish>
  </collection>
</initial>
