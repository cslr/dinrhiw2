<TeXmacs|1.99.12>

<style|<tuple|generic|old-spacing|old-dots>>

<\body>
  <\doc-data|<doc-title|Non-linear principal component
  analysis>|<doc-author|<author-data|<author-name|Tomas Ukkonen,
  tomas.ukonen@iki.fi, 2014>>>>
    \;

    \;
  </doc-data|>

  <em|NOTE: This does not work very well..>

  \;

  One approach in non-linear PCA is to try to diagonalize non-linear matrix\ 

  <\with|par-mode|center>
    <math|\<b-G\><rsub|\<b-x\>\<b-x\>>=E<around*|{|<around*|(|g<around*|(|\<b-x\>|)>-E<around*|{|g<around*|(|\<b-x\>|)>|}>|)><around*|(|g<around*|(|\<b-x\>|)>-E<around*|{|g<around*|(|\<b-x\>|)>|}>|)><rsup|H>|}>>,
  </with>

  where <math|g<around*|(|x|)>> function has been chosen to create
  interesting higher order moments. Diagonalization of the matrix means that
  <math|E<around*|{|g<around*|(|x|)>*<rsub|i>g<around*|(|x|)><rsub|j>|}>=E<around*|{|g<around*|(|x|}><rsub|i>*E<around*|{||\<nobracket\>>g<around*|(|x|)><rsub|j>|}>>,
  one consequence of the independence of the <math|x<rsub|i>> and
  <math|x<rsub|j>>.

  Here <math|g<around*|(|\<b-x\>|)>> can be mapping from low dimensions into
  extremely high dimensions <math|g<around*|(|\<b-x\>|)>:\<frak-R\><rsup|D<rsub|1>>\<rightarrow\>\<frak-R\><rsup|D<rsub|2>>>
  where <math|D<rsub|2>\<gg\>D<rsub|1>> where we want to have property that
  our data components are independent. One example of such mapping could be
  <math|g<around*|(|x,y|)>=<around*|[|x,x<rsup|2>,x<rsup|3>,x<rsup|4>,x<rsup|5>\<ldots\>x<rsup|N><rsup|>,y,y<rsup|2>\<ldots\>y<rsup|N>|]>>
  for a two dimensional variable. Now the function
  <math|g<around*|(|\<b-x\>|)>> should have inverse from high dimensional
  space into low dimensional space which can be used for a dimensional
  reduction.

  <\with|par-mode|center>
    <math|\<b-y\>=g<rsup|-1><around*|(|\<b-W\><around*|(|g<around*|(|\<b-x\>|)>-E<around*|{|g<around*|(|\<b-x\>|)>|}>|)>+\<b-b\>|)>>
  </with>

  where <math|\<b-W\>> matrix diagonalizes <math|G<rsub|\<b-x\>*\<b-x\>>>
  matrix. We can use PCA for this, however, the solution is still not unique
  as there are <math|D<rsub|2>>! different ordering of the variables.

  For simplicity, lets first concentrate at the basic case where we have only
  a single variable <math|x> and <math|g<around*|(|x|)>=<around*|[|x,x<rsup|2>,x<rsup|3>\<ldots\>x<rsup|N>|]>>.
  This means vector <math|\<b-y\>=\<b-W\>*g<around*|(|x|)>> is now a series
  of Taylor polynomials,

  <\with|par-mode|center>
    <math|y<around*|(|n|)>=\<b-w\><rsub|n><rsup|T><around*|(|\<b-g\><around*|(|x|)>-E<around*|{|\<b-g\><around*|(|x|)>|}>|)>=<big|sum><rsub|k>w<rsub|n,k>*x<rsup|k>-<big|sum><rsub|k>w<rsub|n,k>*E<around*|{|x<rsup|k>|}>>
  </with>

  NOTE: Interestingly, the top PCA vector weight vector values have
  distinctive smooth shapes meaning that they define \Pregular\Q functions
  that alter the data <math|\<b-x\>> into decorrelated (independent?)
  signals. On ther other hand, the lowest variance PCA vectors contain
  irregular weights (noise) meaning which seem to imply that these weight
  vectors do not define distinctive functions that arises from the data
  somehow.

  This basically means that if we are about to maximize (remaining) variance
  of <math|y<around*|(|n|)>>, then the algorithm should put most of its
  weight to weigh <math|w<rsub|n,D<rsub|2>>> as
  <math|Var<around*|{|x<rsup|k>|}>> has the maximum variance when <math|k> is
  a large (positive) number. Therefore, to regularize our problem, we need to
  normalize our data to have maximum value of <math|\<pm\>1>,
  <math|x<rprime|'>=x/max<around*|(|<around*|\||x|\|>|)>> which prevents
  variance terms from exploding. This also means most of the time
  <math|Var<around*|{|x<rsup|k>|}>\<geqslant\>Var<around*|{|x<rsup|k+1>|}>>
  for larger <math|k>:s meaning that PCA solution which gives the highest
  variance solutions to top vectors is likely to be good diagonalization of
  <math|\<b-G\><rsub|\<b-x\>*\<b-x\>>> as it preserves ordering of
  <math|\<b-g\><around*|(|x|)>> vectors (?).

  How to compute <math|g<rsup|-1><around*|(|\<b-y\>|)>> vector then. Well it
  is clear that there are <with|font-shape|italic|multiple possible>
  inversion functions to <math|\<b-g\><around*|(|x|)>> function. The (all
  possible?) solutions have the form <math|f<around*|(|\<b-a\><rsup|T>\<b-y\>|)>>
  where <math|\<b-a\>> vector sets up a Taylor polynomial
  <math|t<around*|(|x|)>=<big|sum>a<rsub|k>x<rsup|k>> and
  <math|f<around*|(|x|)>> is inverse of <math|t<around*|(|x|)>>. This means
  that solution vector <math|\<b-a\>> must be chosen so that
  <math|t<around*|(|x|)>> always has an inverse. Additionally, the solution
  vector should have be probably chosen so that it is relatively simple (to
  avoid overfitting to data) and maximizes some desirable property of
  <math|y>, for example, maybe by maximizing its kurtosis value. Overall
  formula to remove higher order correlations from variable <math|x>
  (self-independence). Is therefore

  <\with|par-mode|center>
    <math|y=f<around*|(|\<b-a\><rsup|T><around*|(|\<b-W\><around*|(|*\<b-g\><around*|(|x|)>-E<around*|{|g<around*|(|\<b-x\>|)>|}>|)>+\<b-b\>|\<nobracket\>>|\<nobracket\>>>))
  </with>

  \ where <math|\<b-a\>> and <math|\<b-b\>> are parameters of the model and
  <math|\<b-W\>> should be chosen so that it \Ppreserves\Q ordering of the
  variables somehow. Here we want (initially) the simplest possible model so
  we choose <math|\<b-a\>=<around*|[|1,0,0,0,\<ldots\>|]>> meaning that
  <math|t<around*|(|x|)>=x> and inverse of <math|f<around*|(|x|)>> is simply
  <math|x> and the model simplifies to:

  <\with|par-mode|center>
    <math|y=\<b-w\><rsup|T><rsub|1><around*|(|*\<b-g\><around*|(|x|)>-E<around*|{|\<b-g\><around*|(|x|)>|}>|)>+b<rsub|1>>
  </with>

  Here <math|\<b-w\><rsub|1>> is the first eigenvector of the data (or we
  could choose any! one because ordering of the variables (and in diagonal
  terms of <math|D>) in <strong|W> vector can change freely. Here
  <math|b<rsub|1>> on affects mean of the data and can be chosen to set mean
  of the <math|y> to zero. This means that we only have to solve for
  diagonlization matrix <math|\<b-W\>> and try different eigenvectors.

  \;

  Where we now assume we have extremly high dimensional data which has been
  whitened using some approximative PCA method (for example we use block-wise
  FastPCA, maybe just using a single iteration) after which number of
  dimensions are reduced using non-linear transform. Now, this non-linearity
  has a property that it also, at the same time, diagonalizes
  <math|G<rsub|\<b-x\>*\<b-x\>>> without explicitely calculating
  <math|g<around*|(|\<b-x\>|)>> function. Our two-dimensional function is
  easy to invert as one just have to take every <math|N>:th elements from the
  input vector as rest are higher moments. However, this linear mapping is
  not necessarily proper as out example shows.

  <strong|CONTINUE FROM HERE, WHAT PROPERTIES MUST FUNCTION
  <math|g<around*|(|\<b-x\>|)>> HAVE IN ORDER FOR THIS TO WORK?>

  \;

  Assume we have two different datasets <math|\<b-x\>> and <math|\<b-y\>>.
  Then we want to similarize these two datasets so that they have same higher
  order correlations <math|\<b-G\><rsub|\<b-x\>*\<b-x\>>=\<b-G\><rsub|\<b-y\>*\<b-y\>>>.

  <\with|par-mode|center>
    <math|\<b-G\><rsub|\<b-x\>\<b-x\>>=E<around*|{|<around*|(|\<b-g\><around*|(|\<b-x\>|)>-E<around*|{|\<b-g\><around*|(|\<b-x\>|)>|}>|)><around*|(|\<b-g\><around*|(|\<b-x\>|)>-E<around*|{|\<b-g\><around*|(|\<b-x\>|)>|}>|)><rsup|H>|}>>
  </with>

  Then the solution to this is:

  <\with|par-mode|center>
    <math|\<b-z\>=\<b-g\><rsup|-1><around*|(|\<b-G\><rsup|1/2><rsub|\<b-y\>*\<b-y\>>*\<b-W\><around*|(|\<b-g\><around*|(|\<b-x\>|)>-E<around*|{|\<b-g\><around*|(|\<b-x\>|)>|}>|)>|)>>
  </with>

  where <math|\<b-W\>> diagonalizes <math|\<b-G\><rsub|\<b-x\>*\<b-x\>>>,
  because:

  <\with|par-mode|center>
    <math|<with|font-series|bold|math-font-series|bold|G<rsub|z*z>>=E<around*|{|\<b-G\><rsup|1/2><rsub|\<b-y\>*\<b-y\>>*\<b-W\><around*|(|\<b-g\><around*|(|\<b-x\>|)>-E<around*|{|\<b-g\><around*|(|\<b-x\>|)>|}>|)><around*|(|\<b-g\><around*|(|\<b-x\>|)>-E<around*|{|\<b-g\><around*|(|\<b-x\>|)>|}>|)><rsup|H>\<b-W\><rsup|H>\<b-G\><rsup|1/2><rsub|\<b-y\>*\<b-y\>>*|}>>

    <math|\<b-G\><rsub|\<b-z\>*\<b-z\>>=\<b-G\><rsup|1/2><rsub|\<b-y\>*\<b-y\>>\<b-W\>*\<b-G\><rsub|\<b-x\>*\<b-x\>>\<b-W\><rsup|\<b-H\>>\<b-G\><rsup|1/2><rsub|\<b-y\>*\<b-y\>>>
  </with>

  Now we can choose <math|\<b-W\>=\<b-Lambda\><rsub|\<b-x\>><rsup|-1/2>*\<b-X\><rsup|H>>
  where <math|\<b-G\><rsub|\<b-x\>*\<b-x\>>=\<b-X\>\<b-Lambda\><rsub|\<b-x\>>*\<b-X\><rsup|\<b-H\>>>.
  Now the <math|\<b-G\><rsup|1/2><rsub|\<b-y\>*\<b-y\>>=\<b-Y\>\<b-Lambda\><rsup|1/2><rsub|\<b-y\>>\<b-Y\><rsup|H>>
  and is hermitian too. So we have <math|\<b-G\><rsub|\<b-z\>*\<b-z\>>=\<b-G\><rsub|\<b-y\>*\<b-y\>>>.

  <\with|par-mode|center>
    \;
  </with>

  \;

  \;

  Now what is good non-linearity for <math|g<around*|(|x|)>>? It seems that
  <math|g<around*|(|x|)>=sinh<around*|(|x|)>> is rather good giving that it
  is zero preserving, odd function (as long as we don't add the
  <math|E<around*|{|g<around*|(|x|)>|}>> term). And it's Taylor's expansion
  gives only positive coefficients for <math|x<rsup|2k+1>> terms meaning that
  it should \Ppush' positive and negative moments into different directions.

  How to diagonalize <math|\<b-G\><rsub|\<b-x\>\<b-x\>>> then? The solution
  to this question is almost trivial. Solution\ 

  <\with|par-mode|center>
    <math|\<b-y\>=g<rsup|-1><around*|(|\<b-W\><around*|(|\<b-x\>-E<around*|{|\<b-x\>|}>|)>+\<b-b\><rprime|'>|)>>
  </with>

  diagonalizes <math|\<b-G\><rsub|\<b-x\>\<b-x\>>> matrix when <math|\<b-W\>>
  diagonalizes the <math|E<around*|{|<around*|(|\<b-x\>-E<around*|{|\<b-x\>|}>|)><around*|(|\<b-x\>-E<around*|{|\<b-x\>|}>|)><rsup|H>|}>>
  matrix.\ 

  <\with|par-mode|center>
    <math|E<around*|{|g<around*|(|\<b-y\>|)>|}>=E<around*|{|\<b-W\><around*|(|\<b-x\>-E<around*|{|\<b-x\>|}>|)>+\<b-b\><rprime|'>|}>=\<b-b\><rprime|'>>
  </with>

  <\with|par-mode|center>
    <math|\<b-G\><rsub|\<b-y\>*\<b-y\>>=><math|E<around*|{|<around*|(|g<around*|(|\<b-y\>|)>-E<around*|{|g<around*|(|\<b-y\>|)>|}>|)><around*|(|g<around*|(|\<b-y\>|)>-E<around*|{|g<around*|(|\<b-y\>|)>|}>|)><rsup|H>|}>>

    <\math>
      =E<around*|{|<around*|(|\<b-W\><around*|(|\<b-x\>-E<around*|{|\<b-x\>|}>|)>+\<b-b\><rprime|'>-\<b-b\><rprime|'>|)>*<around*|(|\<b-W\><around*|(|\<b-x\>-E<around*|{|\<b-x\>|}>|)>+\<b-b\><rprime|'>-\<b-b\><rprime|'>|)><rsup|H>|}>

      =\<b-W\>*E<around*|{|<around*|(|\<b-x\>-E<around*|{|\<b-x\>|}>|)><around*|(|\<b-x\>-E<around*|{|\<b-x\>|}>|)><rsup|H>|}>*\<b-W\><rsup|H>=\<b-D\>
    </math>
  </with>

  Now, we can see parallel between diagonalizing
  <math|\<b-G\><rsub|\<b-x\>*\<b-x\>>> matrix and a single layer of a neural
  network <math|\<b-y\>=f<around*|(|\<b-x\>|)>>. Inverse
  <math|g<rsup|-1><around*|(|x|)>> of <math|sinh<around*|(|x|)>> is
  <math|asinh<around*|(|x|)>> - a function that is close to
  <math|tanh<around*|(|x|)>> or sigmoidal functions that have been proposed
  as non-linearities for neural networks. (Additionally,
  <math|sinh<around*|(|x|)>> has only positive coefficients in taylor
  expansion and \ only odd terms: <math|x<rsup|2k+1>> meaning that it is
  \Psuper-odd\Q function as each term of polynomial form is a odd function).\ 

  <\with|par-mode|center>
    <math|\<b-y\>=f<around*|(|\<b-V\>*\<b-x\>+\<b-b\>|)>>

    <\math>
      \<b-V\>=\<b-W\>\<nocomma\>\<nocomma\>,\<b-b\>=\<b-b\><rprime|'>-\<b-W\>*E<around*|{|\<b-x\>|}>
    </math>
  </with>

  One the other hand, <math|\<b-b\>> is free parameter of the function fixing
  expectation <math|E<around*|{|g<around*|(|\<b-y\>|)>|}>> to any wanted
  val.ue. How <math|\<b-b\>> should be chosen then? It is clear that
  selection of the be <math|\<b-b\>> controls distribution of
  <math|p<around*|(|\<b-y\>|)>> and therefore directly the value of
  <math|E<around*|{|g<around*|(|\<b-y\>|)>|}>>. We have decided that
  <math|\<b-b\>> is free parameter of optimization but <math|\<b-b\>=0> might
  make sense too since <math|g<around*|(|\<b-x\>|)>> is odd function and data
  is distributed around mean zero.

  <with|font-shape|italic|NOTE: general solution to the diagonalization
  problem is <math|\<b-y\>=g<rsup|-1><around*|(|\<b-V\>*f<around*|(|\<b-x\>|)>|)>>>,
  where <math|\<b-V\>> diagonalizes <math|E<around*|{|f<around*|(|\<b-x\>|)>*f<around*|(|\<b-x\>|)><rsup|H>|}>>
  matrix and we can choose <math|f<around*|(|x|)>=g<around*|(|x|)>>.

  The whole algorithm for training neural networks is then to stimulate
  neural network, collect input samples for each layer and sequentially
  calculate PCA solution for the inputs and use <math|asinh<around*|(|x|)>>
  non-linearity instead of more standard ones.\ 

  The beauty of the solution comes from that it fits to the previous neural
  network structures and jointly diagnonalizes
  <math|E<around*|{|\<b-x\>*\<b-x\><rsup|H>|}>> and
  <math|E<around*|{|g<around*|(|\<b-x\>|)>g<around*|(|\<b-x\>|)><rsup|H>|}>>
  matrixes and tells how to integrate ICA solutions into non-linear neural
  networks.

  <em|NOTE: PCA/ICA solutions should have variance of 2.0 or something as
  <math|asinh<around*|(|x|)>> has scaling that makes values between 2.0-4.0
  work the best.>

  <strong|Additions>

  Now in order to diagonalize <math|\<b-G\><rsub|\<b-x\>\<b-x\>>> we then
  need to solve for <math|\<b-W\>> matrix that computes PCA from the data. In
  practice, when we are learning data (optimizing neural network), we want to
  update <math|\<b-W\><rsub|g>> matrix according to gradient descent or other
  optimization method so that it is at the same time calculates PCA from
  data. Now, given input data <math|<around*|{|\<b-x\><rsub|i>|}>> for each
  layer, the PCA solution for the whitening matrix is
  <math|\<b-W\><rprime|'>=\<b-D\>*\<b-Z\>*\<b-W\>=\<b-D\>*\<b-Z\>*\<b-Lambda\><rsup|-0.5>\<b-X\><rsup|T>>,
  where <math|\<b-Sigma\><rsub|\<b-x\>*\<b-x\>>=\<b-X\>*\<b-Lambda\>*\<b-X\><rsup|T>>
  , <math|\<b-Z\>> is a freely chosen rotation matrix and <math|\<b-D\>> is a
  diagonal scaling matrix. We now have a linear problem:

  <\with|par-mode|center>
    <math|\<b-W\><rsub|g>=\<b-D\>*\<b-Z\>*\<b-W\>>
  </with>

  \ which is trivially solved by <math|\<b-D\>*\<b-Z\><rprime|'><rsup|>=\<b-W\><rsub|g>*\<b-W\><rsup|-1>>
  but this is not a rotation and breaks PCA property. We want to solve for a
  optimal rotation <math|\<b-Z\>> and scaling <math|\<b-D\>>,
  <math|\<b-Z\><rsup|T>\<b-Z\>=\<b-I\>>. However, this seems to be
  computationally very difficult to do so we instead try to solve
  <math|\<b-Z\>> and <math|\<b-D\>> separatedly. So we initially solve for
  <math|\<b-Z\>> which is as close as possible to <math|\<b-Z\><rprime|'>>
  matrix and assume <math|\<b-D\>=\<b-I\>>. This is a variant of
  <with|font-shape|italic|orthogonal procrustes problem>:

  <\with|par-mode|center>
    <math|min<rsub|\<b-Z\>><around*|\<\|\|\>|\<b-W\><rsub|g>*-\<b-Z\>*\<b-W\>|\<\|\|\>><rsup|2><rsub|F>>
  </with>

  for <math|\<b-Z\>> which will solve PCA solution which keeps weights as
  close as possible towards gradient descent solution (moving towards minimum
  error) of the matrix for each optimization step. This can be solved by
  computing SVD as described in [1]:

  <\with|par-mode|center>
    <math|<around*|\<\|\|\>|\<b-W\><rsub|g>\<b-W\><rsup|-1>-\<b-Z\>|\<\|\|\>><rsub|F><rsup|2>=trace<around*|(|<around*|(|\<b-W\><rsub|g>\<b-W\><rsup|-1>-\<b-Z\>|)><rsup|H><around*|(|\<b-W\><rsub|g>\<b-W\><rsup|-1>-\<b-Z\>|)>|)>>

    <\math>
      trace<around*|(|\<b-W\><rsup|H><rsub|g>\<b-W\><rsup|-1>\<b-W\><rsup|-H>*\<b-W\><rsub|g>|)>+trace<around*|(|\<b-I\>|)>-2*trace<around*|(|<around*|(|\<b-D\>*\<b-Z\>|)>*<rsup|H>\<b-W\><rsub|g>*\<b-W\><rsup|-1>|)>
    </math>
  </with>

  So we need to maximize for <math|trace<around*|(|\<b-Z\><rsup|H>*\<b-W\><rsub|g>*\<b-W\><rsup|-1>|)>=trace<around*|(|\<b-Z\><rsup|H>*\<b-U\>*\<b-S\>*\<b-V\><rsup|H>|)>>
  when <math|\<b-W\><rsub|g>*\<b-W\><rsup|-1>=\<b-U\>\<b-S\>\<b-V\><rsup|H>>
  and we have\ 

  <\with|par-mode|center>
    <math|trace<around*|(|\<b-V\><rsup|H>\<b-Z\><rsup|H>\<b-U\>*\<b-S\>*|)>=trace<around*|(|\<b-Y\>*\<b-S\>|)>>
  </with>

  where <math|\<b-Y\><rsup|H>\<b-Y\>=\<b-I\>> and <math|\<b-Y\>> is
  orthonormal rotation matrix. Now, <math|\<b-S\>> is diagonal matrix meaning
  that all of its \Pmass\Q is on the diagonal and any further rotations will
  only move \Pvariance\Q away from the diagonal meaning that optimum
  <math|\<b-Y\>=\<b-V\><rsup|H>\<b-Z\><rsup|H>\<b-U\>=\<b-I\>> and
  <math|\<b-Z\>=\<b-U\>*\<b-V\><rsup|H>> is the solution to the problem.
  Furthermore, <math|\<b-Z\>> is real because
  <math|\<b-W\><rsub|g>\<b-W\><rsup|-1>> is real and SVD decomposition is
  real for real valued matrixes. Additionally because:\ 

  <\with|par-mode|center>
    <math|\<b-W\><rsub|g>*\<b-W\><rsup|-1>=\<b-W\><rsub|g>*\<b-X\>*\<b-Lambda\><rsup|0.5>>
  </with>

  So we don't even have to calculate inverse matrix.

  Next, after solving for the optimally rotated matrix
  <math|\<b-Q\>=\<b-Z\>*\<b-W\>>, we solve for <math|\<b-D\>> separatedly as
  this step can only improve value of the frobenius norm:\ 

  <\with|par-mode|center>
    <math|min<rsub|\<b-D\>><around*|\<\|\|\>|\<b-W\><rsub|g>-\<b-D\>*\<b-Q\>|\<\|\|\>>=min<rsub|d<rsub|1>\<ldots\>d<rsub|dim<around*|(|\<b-D\>|)>>><big|sum><rsup|i=dim<around*|(|\<b-D\>|)>><rsub|i=1><around*|\<\|\|\>|\<b-w\><rsup|T><rsub|g,i>-d<rsub|i>*\<b-q\><rsub|i><rsup|T>|\<\|\|\>><rsup|2>>
  </with>

  where each vector is a row vector from <math|\<b-W\><rsub|g>> and
  <strong|Q> matrixes. It is easy to minimize these terms:

  <\with|par-mode|center>
    <em|<math|<rsub|>\<xi\><rsub|i><around*|(|d<rsub|i>|)>=>><math|<around*|\<\|\|\>|\<b-w\><rsub|g,i>-d<rsub|i>*\<b-q\><rsub|i>|\<\|\|\>><rsup|2>=<around*|\<\|\|\>|\<b-w\><rsub|g,i>|\<\|\|\>><rsup|2>-2*d<rsub|i>\<b-q\><rsup|T><rsub|i>\<b-w\><rsub|g.i>+d<rsup|2><rsub|i><around*|\<\|\|\>|\<b-q\><rsub|i>|\<\|\|\>><rsup|2>>

    <math|d\<xi\><rsub|i><around*|(|d<rsub|i>|)>/d*d<rsub|i>=-><math|2*\<b-q\><rsup|T><rsub|i>\<b-w\><rsub|g,i>+2*d<rsub|i><around*|\<\|\|\>|\<b-q\><rsub|i>|\<\|\|\>><rsup|2>=0>

    <math|d<rsub|i>=<frac|\<b-q\><rsup|T><rsub|i>\<b-w\><rsub|g,i>|<around*|\<\|\|\>|\<b-q\><rsub|i>|\<\|\|\>><rsup|2>>>
  </with>

  For each iteration <math|n>, the algorithm then first calculates
  <math|\<b-W\><rsub|g><around*|(|n|)>> matrix which is \Praw\Q update for
  the next approximation of the <strong|W> but instead calculates the closest
  <math|\<b-W\><rsub|>> that also computes PCA of the data meaning that
  optimization now optimizes/rotates neural network weights in \Pa PCA
  subspace\Q which additionally diagonalizes non-linear ICA matrix.

  <strong|PROBLEMS>

  However, in practice this <em|do not work> very well, gradient descent
  cannot converge anywhere and more complicated methods that use gradient DO
  NOT work when <math|\<b-W\>> is approximated after the update with
  \PPCArization\Q. Now we have <math|\<b-W\><rsub|g><around*|(|n+1|)>=\<b-W\><rsub|g><around*|(|n|)>+d*\<b-W\><rsub|g>>
  and we want to directly calculate <em|gradient> that rotates matrix
  optimally.

  However, this regularizes neural network to work with \Pindependent
  components\Q. Instead of freely chosen <math|\<b-W\>> we now have rotation
  <math|\<b-Z\>> and bias terms <math|\<b-b\>> which directly controls next
  layers input <math|E<around*|{|g<around*|(|\<b-y\>|)>|}>>. Additionally, if
  we force <math|E<around*|{|g<around*|(|\<b-y\>|)>|}>> to be zero, we have
  <math|\<b-b\>=0> and networks only free parameters are rotations and
  scalings <math|\<b-Z\><rsub|i>> and <math|\<b-D\><rsub|i>> per layer.

  <strong|ULTRA-DEEP RANDOM SEARCH METHOD>

  One approach to the optimization problem
  <math|\<b-p\>=<around*|{|\<b-Z\><rsub|i>,\<b-D\><rsub|i>,\<b-b\><rsub|i>|}><rsub|i>>
  could be to directly construct neural network by choosing parameters
  <math|\<b-p\>> randomly (very high parallelization) and then adding final
  layer which optimizes parameters to zero. However, if we do not use
  <math|\<b-Z\><rsub|i>>, either by using ICA to fix the rotation or just use
  PCA without rotation, then the number of parameters is further reduced to
  <math|\<b-p\>=<around*|{|\<b-d\><rsub|i>,\<b-b\><rsub|i>|}><rsub|i>> which
  mean that number of optimized parameters in search space grows now only
  linearly to the number of neurons <math|O<around*|(|n|)>> and we can create
  ultra deep neuronal networks by always starting from input data, adding new
  layers and immediately forgetting matrix parameters and only keeping
  vectors <math|\<b-p\>=<around*|{|\<b-d\><rsub|i>,\<b-b\><rsub|i>|}><rsub|i>>,
  because it is always possible to reconstruct PCA matrixes (from the exactly
  same training data which must be stored together with parameters).

  The <math|O<around*|(|n|)>> property means that we can use full memory of
  the computer for a single layer computation and create at the same time
  ultra-wide networks (as long as PCA can be computed from the data - for
  this we might want to use fast approximate methods).

  With direct method, one tries different combinations of <math|\<b-p\>>
  until solution improves and then adds extra layer and stores the
  parameters. With random generation of parameters one collects goodness
  values for each parameter <math|<around*|(|\<b-p\>,goodness|)>>. It can
  then make sense to try to do meta learning and with for a linear model
  looking for the better solutions:

  <\with|par-mode|center>
    <math|min<rsub|\<b-w\>>E<around*|{|<around*|\<\|\|\>|g-<around*|(|\<b-a\><rsup|T>\<b-p\>+b|)>|\<\|\|\>>|}>>
  </with>

  \ and use the solution to find for more parameters that might improve
  solution.

  \;

  \;

  \;

  <strong|Final layer linear optimization>

  \;

  Solution to a linear optimization problem \ <math|min
  E<around*|{|<frac|1|2><around*|\<\|\|\>|\<b-y\>-<around*|(|\<b-A\>\<b-x\>+\<b-b\>|)>|\<\|\|\>><rsup|2>|}>>
  is:

  <\with|par-mode|center>
    <math|\<b-A\>=\<b-C\><rsup|T><rsub|\<b-x\>*\<b-y\>>*\<b-C\><rsup|-1><rsub|\<b-x\>*\<b-x\>>>

    <math|\<b-b\>=\<b-mu\><rsup|><rsub|\<b-y\>>-\<b-A\>\<b-mu\><rsub|\<b-x\>>>
  </with>

  Proof:

  min <math|e<around*|(|\<b-a\>,b|)>=E<rsub|x*y><around*|{|0.5*<around*|(|y-\<b-a\><rsup|T>\<b-x\>-b|)><rsup|2>|}>>

  <\math>
    \<nabla\><rsub|\<b-a\>>e<around*|(|\<b-a\>,b|)>=E<rsub|x*y><around*|{|<around*|(|y-\<b-a\><rsup|T>\<b-x\>-b|)>\<b-x\>|}>=0

    \<nabla\><rsub|b>e<around*|(|\<b-a\>,b|)>=E<rsub|x*y><around*|{|-<around*|(|y-\<b-a\><rsup|T>\<b-x\>-b|)>|}>=0

    \;

    E<rsub|x*y><around*|{|\<b-x\>*\<b-x\><rsup|T>|}>\<b-a\>=E<rsub|x*y><around*|{|\<b-x\>*<around*|(|y-b|)>|}>

    \<b-mu\><rsub|\<b-x\>>\<mu\><rsub|y>=\<b-mu\><rsub|\<b-x\>>\<b-mu\><rsup|T><rsub|\<b-x\>>\<b-a\>+b*\<b-mu\><rsub|\<b-x\>>

    \;

    \<b-C\><rsub|\<b-x\>*\<b-x\>>\<b-a\>=E<rsub|x*y><around*|{|\<b-x\>*y|}>-\<b-mu\><rsub|\<b-x\>>\<mu\><rsub|y>

    \;

    \<b-C\><rsub|\<b-x\>*\<b-x\>>*\<b-a\>=\<b-r\><rsub|\<b-x\>y>-*\<b-mu\><rsub|\<b-x\>>\<mu\><rsub|y>

    <around*|[|\<b-R\><rsub|\<b-x\>*\<b-x\>>\<b-a\><rsub|1>,\<b-R\><rsub|\<b-x\>*\<b-x\>>\<b-a\><rsub|2>\<ldots\>\<b-R\><rsub|\<b-x\>*\<b-x\>>\<b-a\><rsub|N>|]>=<around*|[|\<b-r\><rsub|\<b-x\>y<rsub|1>>,\<b-r\><rsub|\<b-x\>y<rsub|2>>\<ldots\>\<b-r\><rsub|\<b-x\>y<rsub|N>>|]>

    \<b-A\>=\<b-C\><rsub|\<b-x\>*\<b-x\>><rsup|-1>*E<around*|{|<around*|(|\<b-x\>-\<b-mu\><rsub|\<b-x\>>|)><around*|(|\<b-y\>-\<b-mu\><rsub|\<b-y\>>|)><rsup|T>|}>

    \<b-A\>=\<b-C\><rsup|-1><rsub|\<b-x\>*\<b-x\>>*\<b-C\><rsub|\<b-x\>*\<b-y\>>

    \;

    \<mu\><rsub|y>=\<b-mu\><rsup|T><rsub|\<b-x\>>\<b-a\>+b

    \<b-mu\><rsup|T><rsub|\<b-y\>>=\<b-mu\><rsup|T><rsub|\<b-x\>>\<b-A\>+\<b-b\><rsup|T>

    \<b-b\>=\<b-mu\><rsup|><rsub|\<b-y\>>-\<b-A\><rsup|T>\<b-mu\><rsub|\<b-x\>>
  </math>

  \;

  But for robust optimization we want to add regularizer to restrict elements
  of <math|\<b-a\>> vector when the problem is ill-defined.

  <\with|par-mode|center>
    <math|min<rsub|\<b-a\>> E<around*|{|<frac|1|2><around*|\<\|\|\>|y-<around*|(|\<b-a\><rsup|T>\<b-x\>+b|)>|\<\|\|\>><rsup|2>|}>+\<lambda\><frac|1|2><around*|\<\|\|\>|\<b-a\>|\<\|\|\>><rsup|2>>
  </with>

  \;

  <\math>
    \<nabla\><rsub|\<b-a\>>e<around*|(|\<b-a\>,b|)>=E<rsub|x*y><around*|{|<around*|(|y-\<b-a\><rsup|T>\<b-x\>-b|)>\<b-x\>|}>+\<lambda\>*\<b-a\>=0

    \<nabla\><rsub|b>e<around*|(|\<b-a\>,b|)>=E<rsub|x*y><around*|{|-<around*|(|y-\<b-a\><rsup|T>\<b-x\>-b|)>|}>=0

    \;

    E<rsub|x*y><around*|{|\<b-x\>*\<b-x\><rsup|T>+\<lambda\>\<b-I\>|}>\<b-a\>=E<rsub|x*y><around*|{|\<b-x\>*<around*|(|y-b|)>|}>

    \<b-mu\><rsub|\<b-x\>>\<mu\><rsub|y>=\<b-mu\><rsub|\<b-x\>>\<b-mu\><rsup|T><rsub|\<b-x\>>\<b-a\>+b*\<b-mu\><rsub|\<b-x\>>

    \;

    \<b-C\><rsub|\<b-x\>*\<b-x\>>\<b-a\>=E<rsub|x*y><around*|{|\<b-x\>*y|}>-\<b-mu\><rsub|\<b-x\>>\<mu\><rsub|y>

    \;

    \<b-C\><rsub|\<b-x\>*\<b-x\>>*\<b-a\>=\<b-r\><rsub|\<b-x\>y>-*\<b-mu\><rsub|\<b-x\>>\<mu\><rsub|y>

    <around*|[|\<b-R\><rsub|\<b-x\>*\<b-x\>>\<b-a\><rsub|1>,\<b-R\><rsub|\<b-x\>*\<b-x\>>\<b-a\><rsub|2>\<ldots\>\<b-R\><rsub|\<b-x\>*\<b-x\>>\<b-a\><rsub|N>|]>=<around*|[|\<b-r\><rsub|\<b-x\>y<rsub|1>>,\<b-r\><rsub|\<b-x\>y<rsub|2>>\<ldots\>\<b-r\><rsub|\<b-x\>y<rsub|N>>|]>

    \<b-A\>=\<b-C\><rsub|\<b-x\>*\<b-x\>><rsup|-1>*E<around*|{|<around*|(|\<b-x\>-\<b-mu\><rsub|\<b-x\>>|)><around*|(|\<b-y\>-\<b-mu\><rsub|\<b-y\>>|)><rsup|T>|}>

    \<b-A\>=\<b-C\><rsup|-1><rsub|\<b-x\>*\<b-x\>>*\<b-C\><rsub|\<b-x\>*\<b-y\>>

    \;

    \<mu\><rsub|y>=\<b-mu\><rsup|T><rsub|\<b-x\>>\<b-a\>+b

    \<b-mu\><rsup|T><rsub|\<b-y\>>=\<b-mu\><rsup|T><rsub|\<b-x\>>\<b-A\>+\<b-b\><rsup|T>

    \<b-b\>=\<b-mu\><rsup|><rsub|\<b-y\>>-\<b-A\><rsup|T>\<b-mu\><rsub|\<b-x\>>
  </math>

  \;

  [1] A Flexible New Technique for Camera Calibration. Zhengyou Zhang. 1998.
  Technical Report

  MSR-TR-98-71.

  \;

  <doc-data|<doc-title|Stacked Non-Linear ICA>>

  Because my previous work about non-linear ICA seemed to generate some
  interesting results I'm trying to document and redo that work from C++
  source code that I still have. I think those results might be interesting
  to generate higher order features as it sometimes seemed to generate on/off
  higher order features that can be useful.

  Non-linear ICA layer:

  <\enumerate-numeric>
    <item>Calculate PCA of the input data and whiten it, set variance to be
    correct for the non-linearity used (1 for tanh, 2-4 for sinh?)

    <item>Diagonalize non-linear <math|\<b-G\>=E<around*|{|g<around*|(|\<b-x\>|)>*g<around*|(|\<b-x\>|)><rsup|T>|}>>
    matrix by using transformation <math|g<rsup|-1><around*|(|\<b-W\>*g<around*|(|\<b-x\>|)>|)>>
    where <math|\<b-W\>> diagonalizes <math|\<b-G\>>

    <item>Go again to step 1 to extract more independent non-linear ICA
    solutions
  </enumerate-numeric>

  The problem is that non-linear ICA solutions are NOT unique. There are
  endless number of non-linear ICA solutions all which depend on
  non-linearity or other calculation technique used.

  \;

  Another technique that could be used to try to cause
  <math|E<around*|{|\<b-x\>*g<around*|(|\<b-W\>*\<b-x\>|)><rsup|T>|}>> matrix
  to be zero as this will then cause FastICA algorithm to converge meaning
  that <math|\<b-W\>> matrix now causes data to be in the form of ICA
  solutions. Additionally, we can interprete
  <math|\<b-y\>=g<around*|(|\<b-W\>\<b-x\>|)>> to be a single layer of neural
  network meaning that the next layer is now decorrelated from the inputs of
  the previous layer.

  <strong|ICA DERIVATION>

  \;

  <\math>
    \<b-w\><rsub|n+1>=E<around*|{|\<b-x\>g<around*|(|\<b-w\><rsup|T>\<b-x\>|)>|}>-E<around*|{|g<rprime|'><around*|(|\<b-w\><rsup|T>\<b-x\>|)>|}><with|font-series|bold|\<b-w\><rsub|n>>
  </math>

  <math|g<rprime|'><around*|(|u|)>=exp<around*|(|-u<rsup|2>/2|)>-u<rsup|2>*exp<around*|(|-u<rsup|2>/2|)>*=<around*|(|1-u<rsup|2>|)>*exp<around*|(|-u<rsup|2>/2|)>>

  Alternative non-linearity is:

  <math|g<around*|(|u|)>=u<rsup|3>,g<rprime|'><around*|(|u|)>=3*u<rsup|2>>

  \;

  \;

  Assume our optimized function is (we have used Lagrange multiplication
  method)

  <math|f<around*|(|\<b-w\>|)>=E<rsub|\<b-x\>><around*|{|G<around*|(|\<b-w\><rsup|T>\<b-x\>|)>|}>+\<lambda\><around*|(|<around*|\<\|\|\>|\<b-w\>|\<\|\|\>><rsup|2>-1|)>>

  <math|\<nabla\>f<around*|(|\<b-w\>|)>=E<rsub|\<b-x\>><around*|{|\<b-x\>*g<around*|(|\<b-w\><rsup|T>\<b-x\>|)>|}>+\<lambda\>\<b-w\>=0>

  It is also possible to solve for the Lagrange multiplicator here. If we
  multiply both sides with optimum <math|\<b-w\><rsub|0><rsup|T>> we have:\ 

  <math|\<lambda\>=-E<rsub|\<b-x\>><around*|{|\<b-w\><rsub|0><rsup|T>\<b-x\>*g<around*|(|\<b-w\><rsub|0><rsup|T>\<b-x\>|)>|}>>

  Furthermore we can also calculate Hessian matrix for the equation:

  <math|H*f<around*|(|\<b-w\>|)>=E<rsub|\<b-x\>><around*|{|\<b-x\>*\<b-x\>*<rsup|T>g<rprime|'><around*|(|\<b-w\><rsup|T>\<b-x\>|)>|}>+\<lambda\>*\<b-I\>>
  and this IF data is independent and sphered approximation has been made
  that \ <math|H*f<around*|(|\<b-w\>|)>=E*<rsub|\<b-x\>><around*|{|\<b-x\>*\<b-x\><rsup|T>|}>E<rsub|\<b-x\>><around*|{|g<rprime|'><around*|(|\<b-w\><rsup|T>\<b-x\>|)>|}>+\<lambda\>*\<b-I\>=><math|E<rsub|\<b-x\>><around*|{|g<rprime|'><around*|(|\<b-w\><rsup|T>\<b-x\>|)>|}>\<b-I\>+\<lambda\>*\<b-I\>>

  Now if we make second order approximation we have equation

  <math|f<around*|(|\<b-w\>+\<Delta\>\<b-w\>|)>=f<around*|(|\<b-w\>|)>+\<nabla\>f<around*|(|\<b-w\>|)><rsup|T>\<Delta\>\<b-w\>+<frac|1|2>\<Delta\>\<b-w\><rsup|T>*H*f<around*|(|\<b-w\>|)>*\<Delta\>\<b-w\>*>

  Now if we derivate this function with respect to <math|\<Delta\>\<b-w\>> we
  can find direction that maximizes the approximated function this gives

  <math|H*f<around*|(|\<b-w\>|)>*\<Delta\>*\<b-w\>=-\<nabla\>f<around*|(|\<b-w\>|)>>

  Injecting previous equations into this gives

  <math|\<b-w\><rsub|n+1>=\<b-w\><rsub|n>-<around*|(|E<rsub|\<b-x\>><around*|{|g<rprime|'><around*|(|\<b-w\><rsub|n><rsup|T>\<b-x\>|)>|}>+\<lambda\>*|)><rsup|-1><around*|(|E<rsub|\<b-x\>><around*|{|\<b-x\>*g<around*|(|\<b-w\><rsub|n><rsup|T>\<b-x\>|)>|}>+\<lambda\>\<b-w\><rsub|n>|)>*>

  Now if we multiply both sides with a scalar
  <math|<around*|(|E<rsub|\<b-x\>><around*|{|g<rprime|'><around*|(|\<b-w\><rsub|n><rsup|T>\<b-x\>|)>|}>+\<lambda\>*|)>>
  we get

  <\math>
    \<b-w\><rsup|*+><rsub|n+1>=\<b-w\><rsub|n><around*|(|E<rsub|\<b-x\>><around*|{|g<rprime|'><around*|(|\<b-w\><rsub|n><rsup|T>\<b-x\>|)>|}>+\<lambda\>*|)>-<around*|(|E<rsub|\<b-x\>><around*|{|\<b-x\>*g<around*|(|\<b-w\><rsub|n><rsup|T>\<b-x\>|)>|}>+\<lambda\>\<b-w\><rsub|n>|)>*

    \<b-w\><rsup|+><rsub|n+1>=E<rsub|\<b-x\>><around*|{|\<b-x\>*g<around*|(|\<b-w\><rsub|n><rsup|T>\<b-x\>|)>|}>-E<rsub|\<b-x\>><around*|{|g<rprime|'><around*|(|\<b-w\><rsub|n><rsup|T>\<b-x\>|)>|}>\<b-w\><rsub|n>\<noplus\>
  </math>

  And normalize <math|<around*|\<\|\|\>||\<nobracket\>>\<b-w\><rsup|+><rsub|n+1><around*|\<\|\|\>|=1|\<nobracket\>>
  for each iteration then we get the FastICA update formula.>

  However, the update formula works considerably BETTER if we choose to use
  plus signs in the update formula. Reason for this is unknown\ 

  <math|\<b-w\><rsup|+><rsub|n+1>=E<rsub|\<b-x\>><around*|{|\<b-x\>*g<around*|(|\<b-w\><rsub|n><rsup|T>\<b-x\>|)>|}>+E<rsub|\<b-x\>><around*|{|g<rprime|'><around*|(|\<b-w\><rsub|n><rsup|T>\<b-x\>|)>|}>\<b-w\><rsub|n>\<noplus\>>

  Additionally, notice that direction of the vector do NOT change when we
  have <math|E<around*|{|\<b-x\>*g<around*|(|\<b-y\>|)><rsup|T>|}>=\<b-0\>>,
  which happens when <math|\<b-x\>> and <math|\<b-y\>=\<b-W\><rsup|T>\<b-x\>>
  are independent because then <math|E<around*|{|\<b-x\>*g<around*|(|\<b-y\>|)>|}>=E<around*|{|\<b-x\>|}>*E<around*|{|g<around*|(|\<b-y\>|)><rsup|T>|}>=\<b-0\>>
  and because we have zero mean data. Therefore, we could try to instead
  diagonalize matrix

  <\math>
    \<b-G\><rsub|\<b-x\>*\<b-x\>>=E<around*|{|<around*|(|\<b-x\>+g<around*|(|\<b-y\>|)>-E<around*|{|\<b-g\><around*|(|\<b-y\>|)>|}>|)><around*|(|\<b-x\>+g<around*|(|\<b-y\>|)>-E<around*|{|\<b-g\><around*|(|\<b-y\>|)>|}>|)><rsup|T>|}>=E<around*|{|\<b-x\>*\<b-x\><rsup|T>|}>+E<around*|{|*\<b-x\>*g<around*|(|\<b-y\>|)><rsup|T>+\<b-g\><around*|(|\<b-y\>|)>\<b-x\><rsup|T>|}>+E<around*|{|g<around*|(|\<b-y\>|)>*g<around*|(|\<b-y\>|)><rsup|T>|}>

    =\<b-I\>+E<around*|{|<around*|(|g<around*|(|\<b-y\>|)>-E<around*|{|\<b-g\><around*|(|\<b-y\>|)>|}>|)><around*|(|g<around*|(|\<b-y\>|)>-E<around*|{|\<b-g\><around*|(|\<b-y\>|)>|}>|)><rsup|T>|}>
  </math>

  But now when <math|\<b-y\>=\<b-W\><rsup|T>\<b-x\>> are independent we have
  <math|E<around*|{|g<around*|(|y<rsub|i>|)>g<around*|(|y<rsub|j>|)>|}>=E<around*|{|g<around*|(|y<rsub|i>|)>|}>E<around*|{|g<around*|(|y<rsub|j>|)>|}>>
  meaning that the second term is also diagonal.

  <with|font-series|bold|What non-linearity to use for the independence
  then?>

  Inverted pseudo-normal distribution probably measures non-gaussianity
  rather well:\ 

  <center|<math|G<around*|(|x|)>=-e<rsup|-<frac|1|2>x<rsup|2>>>>

  And it has rather interesting derivate which we maybe want to use

  <\center>
    <math|g<around*|(|x|)>=x*e<rsup|-<frac|1|2>x<rsup|2>>>

    <math|g<rprime|'><around*|(|x|)>=<around*|(|1-x<rsup|2>|)>*exp<around*|(|-x<rsup|2>/2|)>>
  </center>

  This term has interesting properties that it is close to linear within the
  range of <math|<around*|[|-1,1|]>> and then rapidly goes back to zero
  between <math|<around*|[|1,3|]>> range. This means that the activation
  function ignores too large inputs and wants to keep data within [-2,2]
  range or something.

  \;

  \;
</body>

<initial|<\collection>
</collection>>