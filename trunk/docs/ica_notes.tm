<TeXmacs|1.0.7.18>

<style|generic>

<\body>
  <\doc-data|<doc-title|Neural network independent component space training
  algorithm>|<doc-author|<author-data|<author-name|Tomas Ukkonen,
  tomas.ukonen@iki.fi, 2014>>>>
    \;

    \;
  </doc-data|>

  One approach in ICA is to try to diagonalize non-linear matrix\ 

  <\with|par-mode|center>
    <math|\<b-G\><rsub|\<b-x\>\<b-x\>>=E<around*|{|<around*|(|g<around*|(|\<b-x\>|)>-E<around*|{|g<around*|(|\<b-x\>|)>|}>|)><around*|(|g<around*|(|\<b-x\>|)>-E<around*|{|g<around*|(|\<b-x\>|)>|}>|)><rsup|H>|}>>,
  </with>

  where <math|g<around*|(|x|)>> function has been chosen to create
  interesting higher order moments. Diagonalization of the matrix means that
  <math|E<around*|{|g<around*|(|x<rsub|i>|)>*g<around*|(|x<rsub|j>|)>|}>=E<around*|{|g<around*|(|x<rsub|i>|}>*E<around*|{||\<nobracket\>>g<around*|(|x<rsub|j>|)>|}>>,
  one consequence of the independence of the <math|x<rsub|i>> and
  <math|x<rsub|j>>.

  Now what is good non-linearity for <math|g<around*|(|x|)>>? It seems that
  <math|g<around*|(|x|)>=sinh<around*|(|x|)>> is rather good giving that it
  is zero preserving, odd function (as long as we don't add the
  <math|E<around*|{|g<around*|(|x|)>|}>> term). And it's Taylor's expansion
  gives only positive coefficients for <math|x<rsup|2k+1>> terms meaning that
  it should ``push' positive and negative moments into different directions.

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
  ``super-odd'' function as each term of polynomial form is a odd function).\ 

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
  that all of its ``mass'' is on the diagonal and any further rotations will
  only move ``variance'' away from the diagonal meaning that optimum
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
  <math|\<b-W\><rsub|g><around*|(|n|)>> matrix which is ``raw'' update for
  the next approximation of the <strong|W> but instead calculates the closest
  <math|\<b-W\><rsub|>> that also computes PCA of the data meaning that
  optimization now optimizes/rotates neural network weights in ``a PCA
  subspace'' which additionally diagonalizes non-linear ICA matrix.

  <strong|PROBLEMS>

  However, in practice this <em|do not work> very well, gradient descent
  cannot converge anywhere and more complicated methods that use gradient DO
  NOT work when <math|\<b-W\>> is approximated after the update with
  ``PCArization''. Now we have <math|\<b-W\><rsub|g><around*|(|n+1|)>=\<b-W\><rsub|g><around*|(|n|)>+d*\<b-W\><rsub|g>>
  and we want to directly calculate <em|gradient> that rotates matrix
  optimally.

  However, this regularizes neural network to work with ``independent
  components''. Instead of freely chosen <math|\<b-W\>> we now have rotation
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
  <around*|\<\|\|\>|\<b-y\>-<around*|(|\<b-A\>\<b-x\>+\<b-b\>|)>|\<\|\|\>>>
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

  \;

  [1] A Flexible New Technique for Camera Calibration. Zhengyou Zhang. 1998.
  Technical Report

  MSR-TR-98-71.

  \;
</body>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
  </collection>
</references>