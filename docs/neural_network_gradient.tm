<TeXmacs|1.99.12>

<style|<tuple|generic|old-spacing|old-dots>>

<\body>
  <center|<strong|General Gradient of Neural Network<center|>>>

  <center|<em|tomas.ukkonen@iki.fi>, 2017>

  Backpropagation is a commonly known algorithm for computing the gradient of
  error function which arises when we know target values and the loss, cost
  or error function is one dimensional. Generalizing this to general gradient
  calculation when we seek to find the maximum or minimum value of a neural
  network (thought often ill-fated because of local optimas produced by an
  overfitted neural network) is then important. This is needed, for example,
  when implementing certain reinforcement learning methods.

  Consider a two-layer neural network

  <center|<math|y<around*|(|\<b-x\>|)>=f<around*|(|\<b-W\><rsup|<around*|(|2|)>>*g<around*|(|\<b-W\><rsup|<around*|(|1|)>>\<b-x\>+\<b-b\><rsup|<around*|(|1|)>>|)>+\<b-b\><rsup|<around*|(|2|)>>|)>>>

  The gradients of the final layer are (non-zero terms <math|g<rsub|i>> are
  at the <math|j>:th row, <math|\<b-v\>=\<b-W\><rsup|<around*|(|2|)>>\<b-g\>+\<b-b\><rsup|<around*|(|2|)>>>):

  <\center>
    <math|<frac|\<partial\>*y<around*|(|\<b-x\>|)>|\<partial\>*w<rsub|j*i><rsup|<around*|(|2|)>>>=diag<around*|(|<frac|\<partial\>f<around*|(|\<b-v\>|)>|\<partial\>\<b-v\>>|)>*<matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|g<rsub|i>>>|<row|<cell|0>>>>>>

    <math|<frac|\<partial\>*y<around*|(|\<b-x\>|)>|\<partial\>*b<rsub|j><rsup|<around*|(|2|)>>>=diag<around*|(|<frac|\<partial\>f<around*|(|\<b-v\>|)>|\<partial\>\<b-v\>>|)>*<matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|1>>|<row|<cell|0>>>>>>
  </center>

  The derivation chain-rule can be used to calculate the second (and more
  deep layers' gradients):

  <\center>
    <math|<frac|\<partial\>*y<around*|(|\<b-x\>|)>|\<partial\>*w<rsub|j*i><rsup|<around*|(|1|)>>>=diag<around*|(|<frac|\<partial\>f<around*|(|\<b-x\>|)>|\<partial\><around*|(|\<b-W\><rsup|<around*|(|2|)>>\<b-g\>+\<b-b\><rsup|<around*|(|2|)>>|)>>|)>*<frac|\<partial\><around*|(|\<b-W\><rsup|<around*|(|2|)>>*\<b-g\>+\<b-b\><rsup|<around*|(|2|)>>|)>|\<partial\>\<b-g\>>*<frac|\<partial\>\<b-g\><around*|(|\<b-x\>|)>|\<partial\><around*|(|\<b-W\><rsup|<around*|(|1|)>>\<b-x\>+\<b-b\><rsup|<around*|(|1|)>>|)>>*<frac|\<partial\><around*|(|\<b-W\><rsup|<around*|(|2|)>>\<b-x\>+\<b-b\><rsup|<around*|(|2|)>>|)>|\<partial\>*w<rsup|1><rsub|j*i>>>

    <math|<frac|\<partial\>*y<around*|(|\<b-x\>|)>|\<partial\>*w<rsub|j*i><rsup|<around*|(|1|)>>>=diag<around*|(|<frac|\<partial\>f<around*|(|\<b-x\>|)>|\<partial\><around*|(|\<b-W\><rsup|<around*|(|2|)>>\<b-g\>+\<b-b\><rsup|<around*|(|2|)>>|)>>|)>*\<b-W\><rsup|<around*|(|2|)>>*diag<around*|(|*<frac|\<partial\>\<b-g\><around*|(|\<b-x\>|)>|\<partial\><around*|(|\<b-W\><rsup|<around*|(|1|)>>\<b-x\>+\<b-b\><rsup|<around*|(|1|)>>|)>>|)>*<matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|x<rsub|i>>>|<row|<cell|0>>>>>>

    <math|<frac|\<partial\>*y<around*|(|\<b-x\>|)>|\<partial\>*b<rsup|<around*|(|1|)>><rsub|j>>=diag<around*|(|<frac|\<partial\>f<around*|(|\<b-x\>|)>|\<partial\><around*|(|\<b-W\><rsup|<around*|(|2|)>>\<b-g\>+\<b-b\><rsup|<around*|(|2|)>>|)>>|)>*\<b-W\><rsup|<around*|(|2|)>>*diag<around*|(|*<frac|\<partial\>\<b-g\><around*|(|\<b-x\>|)>|\<partial\><around*|(|\<b-W\><rsup|<around*|(|1|)>>\<b-x\>+\<b-b\><rsup|<around*|(|1|)>>|)>>|)>*<matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|1>>|<row|<cell|0>>>>>>
  </center>

  By analysing the chain rule we can derive generic backpropagation formula
  for the full gradient. Let <math|\<b-v\><rsup|<around*|(|k|)>>> be a
  <math|k>:th layer's local field, <math|\<b-v\><rsup|<around*|(|k|)>>=\<b-W\><rsup|<around*|(|k|)>>f<around*|(|\<b-v\><rsup|<around*|(|k-1|)>>|)>+\<b-b\><rsup|<around*|(|k|)>>>,
  <math|\<b-v\><rsup|<around*|(|0|)>>=\<b-x\>>. Then the local jacobian
  matrices <math|\<b-delta\><rsup|<around*|(|k|)>>> are

  <\center>
    <math|\<b-delta\><rsup|<around*|(|L|)>>=diag<around*|(|<frac|\<partial\>\<b-f\><around*|(|\<b-v\><rsup|<around*|(|L|)>>|)>|\<partial\>\<b-v\><rsup|<around*|(|L|)>>>|)>>

    <math|\<b-delta\><rsup|<around*|(|k-1|)>>=\<b-delta\><rsup|<around*|(|k|)>>*\<b-W\><rsup|<around*|(|k|)>>*diag<around*|(|<frac|\<partial\>\<b-f\><around*|(|\<b-v\><rsup|<around*|(|k-1|)>>|)>|\<partial\>\<b-v\><rsup|<around*|(|k-1|)>>>|)>=diag<around*|(|<frac|\<partial\>\<b-f\><around*|(|\<b-v\><rsup|<around*|(|k-1|)>>|)>|\<partial\>\<b-v\><rsup|<around*|(|k-1|)>>>|)>*\<b-W\><rsup|<around*|(|k|)><rsup|T>>\<b-delta\><rsup|<around*|(|k|)>>>
  </center>

  And network's parameter gradient matrices for each layer are (only
  <math|j>:th element of each row is non-zero):\ 

  <\center>
    <math|<frac|\<partial\>*y<around*|(|\<b-x\>|)>|\<partial\>*w<rsub|j*i><rsup|<around*|(|k|)>>>=\<b-delta\><rsup|<around*|(|k|)>>*<matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|f<around*|(|v<rsub|i><rsup|<around*|(|k-1|)>>|)>>>|<row|<cell|0>>>>>>

    <math|<frac|\<partial\>*y<around*|(|\<b-x\>|)>|\<partial\>*b<rsub|j><rsup|<around*|(|k|)>>>=\<b-delta\><rsup|<around*|(|k|)>><matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|1>>|<row|<cell|0>>>>>>
  </center>

  To test that gradient matrix is correctly computed it can be compared with
  normal squared error calculations (normal backpropagation).

  <center|<math|MSE<around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>=<frac|1|2><around*|\<\|\|\>|y<rsub|i>-y<around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>|\<\|\|\>><rsup|2>>>

  <center|<math|<frac|\<partial\>MSE<around*|(|\<b-w\>|)>|\<partial\>\<b-w\>>=<around*|(|y<around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>-y<rsub|i>|)><rsup|T>*<frac|\<partial\>y<around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>|\<partial\>*\<b-w\>>>>

  <with|font-series|bold|Backpropagation algorithm>

  To implement backpropagation algorithm we can do the forward step as in the
  previous step and save local field values
  <math|\<b-v\>*<rsup|<around*|(|k|)>>>. After this we calculate local error
  \ <math|E<around*|{|MSE<around*|(|\<b-w\>|)>|}>=<frac|1|2><around*|\<\|\|\>|\<b-f\><around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>-\<b-y\>|\<\|\|\>><rsup|2>>
  vector gradients <math|\<b-sigma\><rsup|<around*|(|k|)>>> using gradient
  matrices from the previous section. Notice that by calculating error's
  gradient we don't have to calculate Jacobian matrix for each layer so the
  computation is faster.

  <\padded-center>
    <math|\<b-sigma\><rsup|<around*|(|L|)>>=<wide|<around*|(|\<b-f\><around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>-\<b-y\>|)><rsup|T>|\<bar\>>*\<b-delta\><rsup|<around*|(|L|)>>=\<b-delta\><rsup|<around*|(|L|)>>*<wide|<around*|(|\<b-f\><around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>-\<b-y\>|)>|\<bar\>>>

    <math|\<b-sigma\><rsup|<around*|(|k-1|)>>=\<b-sigma\><rsup|<around*|(|k-1|)>>*\<b-W\><rsup|<around*|(|k|)>>*diag<around*|(|<frac|\<partial\>\<b-f\><around*|(|\<b-v\><rsup|<around*|(|k-1|)>>|)>|\<partial\>\<b-v\><rsup|<around*|(|k-1|)>>>|)>=diag<around*|(|<frac|\<partial\>\<b-f\><around*|(|\<b-v\><rsup|<around*|(|k-1|)>>|)>|\<partial\>\<b-v\><rsup|<around*|(|k-1|)>>>|)>*\<b-W\><rsup|<around*|(|k|)><rsup|<rsup|T>>>\<b-sigma\><rsup|<around*|(|k-1|)>>>
  </padded-center>

  The actual gradient value formulas are the same as in the previous section
  (notice that for complex valued neural networks in MSE minimization you
  need to calculate conjugate of the Jacobian matrix but not the error term).

  <\center>
    <math|<frac|\<partial\>*MSE<around*|(|\<b-w\>|)>|\<partial\>*w<rsub|j*i><rsup|<around*|(|k|)>>>=<wide|\<b-sigma\><rsup|<around*|(|k|)>>*<matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|f<around*|(|v<rsub|i><rsup|<around*|(|k-1|)>>|)>>>|<row|<cell|0>>>>>|\<bar\>>>

    <math|<frac|\<partial\>*MSE<around*|(|\<b-w\>|)>|\<partial\>*b<rsub|j><rsup|<around*|(|k|)>>>=<wide|\<b-sigma\><rsup|<around*|(|k|)>><matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|1>>|<row|<cell|0>>>>>|\<bar\>>>
  </center>

  \;

  Now for complex valued data we have for example in the last layer:

  <math|>

  <\padded-center>
    <math|<frac|\<partial\>*MSE<around*|(|\<b-w\>|)>|\<partial\>*w<rsub|j*i><rsup|<around*|(|L|)>>>=<wide|\<b-sigma\><rsup|<around*|(|L|)>>*<matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|f<around*|(|v<rsub|i><rsup|<around*|(|k-1|)>>|)>>>|<row|<cell|0>>>>>=|\<bar\>><wide|<wide|<around*|(|\<b-f\><around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>-\<b-y\>|)><rsup|T>|\<bar\>>*\<b-delta\><rsup|<around*|(|L|)>>*<matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|f<around*|(|v<rsub|i><rsup|<around*|(|k-1|)>>|)>>>|<row|<cell|0>>>>>|\<bar\>>>

    <math|=<around*|(|\<b-f\><around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>-\<b-y\>|)><rsup|T>*<wide|\<b-delta\><rsup|<around*|(|L|)>><matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|f<around*|(|v<rsub|i><rsup|<around*|(|k-1|)>>|)>>>|<row|<cell|0>>>>>=|\<bar\>><around*|(|\<b-f\><around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>-\<b-y\>|)><rsup|T><wide|<frac|\<partial\>y<around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>|\<partial\>*\<b-w\><rsub|j*i><rsup|<around*|(|L|)>>>|\<bar\>>>.
  </padded-center>

  \;

  \;

  \;

  \;

  <strong|Gradient of neural network <math|\<b-f\><around*|(|\<b-x\>|)>>
  input vector <math|\<b-x\>>.>

  Sometimes also needs gradient with respect to <math|\<b-x\> > and not
  weights parameters <math|\<b-w\>>. This can be calculated using the chain
  rule again. For simplicity, let's consider two-layer case initially.

  <\center>
    <math|\<b-g\><around*|(|\<b-x\>|)>=\<b-f\><around*|(|\<b-W\><rsup|<around*|(|2|)>>*\<b-h\><around*|(|\<b-W\><rsup|<around*|(|1|)>>\<b-x\>+\<b-b\><rsup|<around*|(|1|)>>|)>+\<b-b\><rsup|<around*|(|2|)>>|)>>
  </center>

  The gradient is:

  <\center>
    <math|<frac|\<partial\>\<b-g\><around*|(|\<b-x\>|)>|\<partial\>\<b-x\>>=<frac|\<partial\>\<b-f\><around*|(|\<b-v\><rsup|<around*|(|2|)>>|)>|\<partial\><around*|(|\<b-W\><rsup|<around*|(|2|)>>\<b-h\>+\<b-b\><rsup|<around*|(|2|)>>|)>>*<frac|\<partial\><around*|(|\<b-W\><rsup|<around*|(|2|)>>\<b-h\>+\<b-b\><rsup|<around*|(|2|)>>|)>|\<partial\>\<b-h\>>*<frac|\<partial\>*\<b-h\><around*|(|\<b-v\><rsup|<around*|(|1|)>>|)>|\<partial\><around*|(|\<b-W\><rsup|<around*|(|1|)>>\<b-x\>+\<b-b\><rsup|<around*|(|1|)>>|)>>*<frac|\<partial\><around*|(|\<b-W\><rsup|<around*|(|1|)>>\<b-x\>+\<b-b\><rsup|<around*|(|1|)>>|)>|\<partial\>*\<b-x\>>>

    <math|<frac|\<partial\>\<b-g\><around*|(|\<b-x\>|)>|\<partial\>\<b-x\>>=<frac|\<partial\>\<b-f\><around*|(|\<b-v\><rsup|<around*|(|2|)>>|)>|\<partial\>\<b-v\><rsup|<around*|(|2|)>>>*\<b-W\><rsup|<around*|(|2|)>>*<frac|\<partial\>*\<b-h\><around*|(|\<b-v\><rsup|<around*|(|1|)>>|)>|\<partial\>\<b-v\><rsup|<around*|(|1|)>>>*><math|\<b-W\><rsup|<around*|(|1|)>>>
  </center>

  \ 

  This results into following formula (diag() entries are square matrices
  which diagonal is nonzero):

  <center|<math|\<nabla\><rsub|\<b-x\>>*\<b-g\><around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>=diag<around*|(|\<nabla\><rsub|\<b-v\><rsup|<around*|(|L|)>>>\<b-f\><around*|(|\<b-v\><rsup|<around*|(|L|)>>|)>|)>\<b-W\><rsup|*<around*|(|L|)>>\<ldots\>*diag<around*|(|\<nabla\><rsub|\<b-v\><rsup|<around*|(|2|)>>>\<b-f\><around*|(|\<b-v\><rsup|<around*|(|2|)>>|)>|)>\<b-W\><rsup|<around*|(|2|)>>*diag<around*|(|\<nabla\><rsub|\<b-v\><rsup|<around*|(|1|)>>>\<b-f\><around*|(|\<b-v\><rsup|<around*|(|1|)>>|)>|)>\<b-W\><rsup|<around*|(|1|)>>>>

  \;

  In continuous reinforcement learning, we need to maximize the given
  policy's <math|\<b-mu\>> average <strong|Q>-value
  <math|\<b-Q\><around*|(|\<b-x\>,\<b-mu\><around*|(|\<b-x\>|)>|)>> which
  gradient can be computed by using the chain-rule but there is additionally
  linear pre- and postprocessings <math|\<b-W\>*\<b-x\>+\<b-b\>> in
  <math|\<b-mu\>> and <math|\<b-Q\>> which makes the calculation of gradient
  more complicated.

  <\math>
    \<nabla\><rsub|<around*|(|\<b-W\><rsub|\<b-mu\>>\<b-x\>+\<b-b\><rsub|\<b-mu\>>|)>>\<b-W\><rsub|Q><rprime|'>*\<b-Q\><around*|(|\<b-W\><rsub|Q>*\<b-z\>+\<b-b\><rsub|Q>|)>+\<b-b\><rprime|'><rsub|Q>,\<b-z\>=<around*|[|\<b-x\>,\<b-W\><rprime|'><rsub|\<b-mu\>>*\<b-mu\><around*|(|\<b-W\><rsub|\<b-mu\>>\<b-x\>+\<b-b\><rsub|\<b-mu\>>|)>+\<b-b\><rprime|'><rsub|\<b-mu\>>|]>=

    \<b-W\><rprime|'><rsub|Q>*\<nabla\><rsub|><rsub|>\<b-Q\><around*|(|\<b-W\><rsub|Q>*\<b-z\>+\<b-b\><rsub|Q>|)>\<b-W\><rsup|<around*|(|\<b-mu\>|)>><rsub|Q>\<b-W\><rprime|'><rsub|\<b-mu\>>\<nabla\>\<b-mu\>
  </math>

  But in practice we don't have post-processing for <math|\<b-mu\>> so the
  gradient becomes

  <center|<math|\<b-W\><rprime|'><rsub|Q>*\<nabla\><rsub|>\<b-Q\><around*|(|\<b-W\><rsub|Q>*\<b-z\>+\<b-b\><rsub|Q>|)>\<b-W\><rsup|<around*|(|\<b-mu\>|)>><rsub|Q>\<nabla\>\<b-mu\>>>

  \;

  \V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V\V

  \;

  <strong|Recurrent Neural Networks and Backpropagation (Similar to RTRL)>

  The basic learning algorithm for recurrent neural networks (RNN) is BPTT
  but I use modified RTRL instead. (RTRL - real time recurrent learning).
  This is done by unfolding neural net in time an computing the gradients.
  The recurrent neural network is

  <center|<math|\<b-u\><around*|(|n+1|)>=<matrix|<tformat|<table|<row|<cell|\<b-y\><around*|(|n+1|)>>>|<row|<cell|\<b-r\><around*|(|n+1|)>>>>>>=\<b-f\><around*|(|\<b-x\><around*|(|n|)>,\<b-r\><around*|(|n|)>|)>>>

  The error function to minimize is:

  <center|<math|E<around*|(|N|)>=<frac|1|2><big|sum><rsup|N><rsub|n=1><around*|\<\|\|\>|\<b-d\><around*|(|n+1|)>-\<b-Gamma\><rsub|\<b-y\>>\<b-f\><around*|(|\<b-x\><around*|(|n|)>,\<b-Gamma\><rsub|\<b-r\>>\<b-u\><around*|(|n-1|)><around*|\||\<b-w\>|\<nobracket\>>|)>|\<\|\|\>><rsup|2>>>

  In which <math|\<b-Gamma\>> matrices are used to select
  <math|\<b-y\><around*|(|n|)>> and <math|\<b-r\><around*|(|n|)>> vectors
  from generic output vector and the initial input to feedforward neural
  network is zero <math|\<b-u\><around*|(|0|)>=\<b-0\>>.

  It is possible to calculate gradient of <math|\<b-f\>> using the chain rule

  <center|<math|<frac|\<partial\>E<around*|(|N|)>|\<partial\>\<b-w\>>=<big|sum><rsup|N><rsub|n=0><around*|(|\<b-Gamma\><rsub|\<b-y\>>\<b-f\><around*|(|\<b-x\><around*|(|n|)>,\<b-Gamma\><rsub|\<b-r\>>\<b-u\><around*|(|n-1|)><around*|\||\<b-w\>|\<nobracket\>>|)>-\<b-d\><around*|(|n+1|)>|)><rsup|T>*\<b-Gamma\><rsub|\<b-y\>>*\<nabla\><rsub|\<b-w\>>\<b-f\><around*|(|\<b-x\><around*|(|n|)>,\<b-Gamma\><rsub|\<b-r\>>\<b-u\><around*|(|n-1|)><around*|\||\<b-w\>|\<nobracket\>>|)>>>

  To calculate the gradient <math|\<nabla\><rsub|\<b-w\>>\<b-f\>> one must
  remember that <math|\<b-u\><around*|(|n|)>> now also depends on
  <math|\<b-w\>> resulting into eq:

  <center|<math|\<nabla\><rsub|\<b-w\>>\<b-f\><around*|(|\<b-x\><around*|(|n|)>,\<b-Gamma\><rsub|\<b-r\>>\<b-u\><around*|(|n-1|)>|)>=<frac|\<partial\>\<b-f\>|\<partial\>\<b-w\>>+<frac|\<partial\>\<b-f\>|\<partial\>\<b-r\>>*\<b-Gamma\><rsub|\<b-r\>><frac|\<partial\>\<b-u\><around*|(|n-1|)>|\<partial\>\<b-w\>>*<rsub|>>>

  To further compute gradients we get a generic update rule

  <center|<math|<frac|\<partial\>*\<b-u\><around*|(|n|)>|\<partial\>\<b-w\>>=<frac|\<partial\>\<b-f\>|\<partial\>\<b-w\>>+<frac|\<partial\>\<b-f\>|\<partial\>\<b-r\>>*\<b-Gamma\><rsub|\<b-r\>>*<frac|\<partial\>\<b-u\><around*|(|n-1|)>|\<partial\>\<b-w\>>>>

  The computation of gradients can be therefore bootstrapped by setting
  <math|<frac|\<partial\>\<b-u\><around*|(|0|)>|\<partial\>\<b-w\>>=<frac|\<partial\>\<b-f\><around*|(|\<b-0\>,\<b-0\>|)>|\<partial\>\<b-w\>>>
  and iteratively updating <math|\<b-u\>> gradient while computing the
  current error for the timestep.

  <strong|RNN-RBM>

  RNN-RBM was described on the web to create learn creating \Pmusic\Q by
  using BPTT but I use extended RTRL approach instead.

  (<verbatim|http://danshiebler.com/2016-08-17-musical-tensorflow-part-two-the-rnn-rbm/>)

  In RNN-RBM we have a standard RBM model but RBM's biases
  <math|\<b-a\><around*|(|n+1|)>> and <math|\<b-b\><around*|(|n+1|)>> are
  generated by a recurrent neural network
  <math|<matrix|<tformat|<table|<row|<cell|\<b-y\><around*|(|n+1|)>>>|<row|<cell|\<b-r\><around*|(|n+1|)>>>>>>=\<b-f\><around*|(|\<b-x\><around*|(|n|)>,\<b-r\><around*|(|n|)><around*|\||\<b-w\>|\<nobracket\>>|)>>
  , <math|\<b-y\><around*|(|n+1|)>=<matrix|<tformat|<table|<row|<cell|\<b-a\><around*|(|n+1|)>>>|<row|<cell|\<b-b\><around*|(|n+1|)>>>>>>>
  and visible units (MIDI notes) are fed to be <strong|>inputs of recurrent
  neural network <math|\<b-x\><around*|(|n|)>=\<b-v\><around*|(|n|)>>.

  One can then compute RBM's log-likelihood gradient with respect to
  recurrent neural networks weights <math|\<b-w\>> maximizing probability of
  \Psemi\Q independent MIDI notes observations\ 

  <center|<math|-log<around*|[|p<around*|(|\<b-v\><around*|(|1|)>,\<b-v\><around*|(|2|)>\<ldots\>\<b-v\><around*|(|N|)>|)>|]>\<approx\><big|sum><rsub|n>-log<around*|(|p<around*|(|\<b-v\><rsub|n>|)>|)>>>

  We want to calculate gradient with res<strong|>pect to <math|\<b-w\>> where
  only elements <math|\<b-a\>> and <math|\<b-b\>> depend on <math|\<b-w\>>.

  <center|<math|<frac|\<partial\>-log<around*|(|p<around*|(|\<b-v\><rsub|i>|)>|)>|\<partial\>*\<b-w\>>=<frac|\<partial\>-log<around*|(|p<around*|(|\<b-v\><rsub|i>|)>|)>|\<partial\>\<b-a\>>*<frac|\<partial\>\<b-a\>|\<partial\>\<b-w\>>+<frac|\<partial\>-log<around*|(|p<around*|(|\<b-v\><rsub|i>|)>|)>|\<partial\>\<b-b\>>*<frac|\<partial\>\<b-b\>|\<partial\>\<b-w\>>>>

  This can be rewritten using free-energy
  <math|p<around*|(|\<b-v\>|)>=<frac|1|Z>*e<rsup|-F<around*|(|\<b-v\>|)>>=<frac|1|Z>*<big|sum><rsub|\<b-h\>>e<rsup|-E<around*|(|\<b-v\>,\<b-h\>|)>>>
  and the gradient formula for <math|<frac|\<partial\>-log<around*|(|p<around*|(|\<b-v\>|)>|)>|\<partial\>\<theta\>>=<frac|\<partial\>*F<around*|(|\<b-v\>|)>|\<partial\>\<theta\>>-E<rsub|\<b-v\>><around*|[|<frac|\<partial\>F<around*|(|\<b-v\>|)>|\<partial\>\<theta\>>|]>>:

  <center|<math|<frac|\<partial\>-log<around*|(|p<around*|(|\<b-v\><rsub|i>|)>|)>|\<partial\>*\<b-w\>>=*<around*|(|<frac|\<partial\>*F<around*|(|\<b-v\>|)>|\<partial\>\<b-a\>>-E<rsub|\<b-v\>><around*|[|<frac|\<partial\>F<around*|(|\<b-v\>|)>|\<partial\>\<b-a\>>|]>|)><frac|\<partial\>\<b-a\>|\<partial\>\<b-w\>>+<around*|(|<frac|\<partial\>*F<around*|(|\<b-v\>|)>|\<partial\>\<b-b\>>-E<rsub|\<b-v\>><around*|[|<frac|\<partial\>F<around*|(|\<b-v\>|)>|\<partial\>\<b-b\>>|]>|)><frac|\<partial\>\<b-b\>|\<partial\>\<b-w\>>>>

  \;

  We assume GB-RBM model (<verbatim|see RBM_notes.tm>) with the following
  energy function\ 

  <center|<math|E<rsub|GB><around*|(|\<b-v\>,\<b-h\>|)>=<frac|1|2><around*|(|\<b-v\>-\<b-a\>|)><rsup|T>\<b-Sigma\><rsup|-1><around*|(|\<b-v\>-\<b-a\>|)>-<around*|(|\<b-Sigma\><rsup|-0.5>*\<b-v\>|)><rsup|T>\<b-W\>*\<b-h\>-\<b-b\><rsup|T>\<b-h\>>>

  And we extend RNN to also output/predict variance
  <math|\<b-z\><around*|(|n|)>=log<around*|(|diag<around*|(|\<b-Sigma\><around*|(|n|)>|)>|)>>
  [here we use <math|\<b-z\><around*|(|n|)>> for two different things, one
  for variance parameter of GB-RBM and other for recurrent neural networks
  output]. Our gradients therefore are (<verbatim|see RBM_notes.tm>):

  <math|<frac|\<partial\>*\<b-u\><around*|(|n|)>|\<partial\>\<b-w\>>=<frac|\<partial\>\<b-f\><around*|(|\<b-v\><around*|(|n|)>|)>|\<partial\>\<b-w\>>+<frac|\<partial\>\<b-f\>|\<partial\>\<b-r\>>*\<b-Gamma\><rsub|\<b-r\>>*<frac|\<partial\>\<b-u\><around*|(|n-1|)>|\<partial\>\<b-w\>>>,
  <math|\<b-u\><around*|(|n|)>=<matrix|<tformat|<table|<row|<cell|\<b-a\><around*|(|n|)>>>|<row|<cell|\<b-b\><around*|(|n|)>>>|<row|<cell|\<b-z\><around*|(|n|)>>>|<row|<cell|\<b-r\><around*|(|n|)>>>>>>>

  <math|<frac|\<partial\>\<b-a\>|\<partial\>\<b-w\>>=\<b-Gamma\><rsub|\<b-a\>><frac|\<partial\>*\<b-u\><around*|(|n|)>|\<partial\>\<b-w\>>>

  <math|<frac|\<partial\>\<b-b\>|\<partial\>\<b-w\>>=\<b-Gamma\><rsub|\<b-b\>><frac|\<partial\>*\<b-u\><around*|(|n|)>|\<partial\>\<b-w\>>>

  <math|<frac|\<partial\>\<b-z\>|\<partial\>\<b-w\>>=\<b-Gamma\><rsub|\<b-z\>><frac|\<partial\>*\<b-u\><around*|(|n|)>|\<partial\>\<b-w\>>>

  <math|<frac|\<partial\>F|\<partial\>\<b-a\>>=-\<b-Sigma\><rsup|-1><around*|(|\<b-v\>-\<b-a\>|)>>

  <math|<frac|\<partial\>F|\<partial\>\<b-b\>>=-sigmoid<around*|(|\<b-W\><rsup|T>\<b-Sigma\><rsup|-1/2>\<b-v\>+\<b-b\>|)>>

  <math|<frac|\<partial\>F|\<partial\>\<b-z\><rsub|i>>=-<frac|1|2><around*|(|v<rsub|i>-a<rsub|i>|)><rsup|2>*e<rsup|-z<rsub|i>>+<frac|1|2>e<rsup|-z<rsub|i>/2>*v<rsub|i>*<big|sum><rsub|<rsup|>j>w<rsub|i*j>*sigmoid<around*|(|\<b-W\><rsup|T>\<b-Sigma\><rsup|-1/2>\<b-v\>+\<b-b\>|)><rsub|j>>

  <math|<frac|\<partial\>F|\<partial\>\<b-W\>>=><math|-sigmoid<around*|(|\<b-W\><rsup|T>\<b-Sigma\><rsup|-1/2>\<b-v\>+\<b-b\>|)>*<around*|(|\<b-Sigma\><rsup|-1/2>*\<b-v\>|)><rsup|T>>

  When using these gradients it is important to remember that, in general,
  one must update variance terms <math|\<b-z\>> independently from other
  parameters or the GB-RBM doesn't converge. Initially, however, I will make
  an attempt to learn both variance <math|\<b-z\>>, <math|\<b-a\>> and
  <math|\<b-b\>> \ because they now all depend on common weight vector
  parameter <math|\<b-w\>>. Pseudocode for RNN-RBM optimization:

  1. randomly initialize <math|\<b-W\>> and <math|\<b-w\>> using small
  values.

  2. Calculate parameters <math|\<b-a\>>, <math|\<b-b\>> and <math|\<b-z\>>
  using current RNN (initially with zero recurrent input parameters including
  previous step's visible MIDI notes <math|\<b-v\><around*|(|n-1|)>>)

  3. Use RBM and CD-k to calculate <math|<around*|(|\<b-v\>,\<b-h\>|)>>
  parameters for input sample(s) and calculate contrastive divergence samples
  in order to calculate gradient of free energy <math|\<nabla\>\<b-F\>>.

  4. Calculate the gradient of the recurrent neural network
  <math|\<nabla\><rsub|\<b-w\>>\<b-u\><around*|(|n|)>> and use
  <math|\<nabla\>\<b-F\>> to calculate gradient of the probability
  <math|p<around*|(|\<b-v\>|)>> with respect to <math|\<b-W\>> and
  <math|\<b-w\>>.

  5. Repeat steps 2-4 for each song <math| i> (time series) of visible notes
  <math|<around*|{|\<b-v\><rsub|i><around*|(|n|)>|}>> and use the sum of all
  songs gradients to move parameters <math|\<b-W\>> and <math|\<b-w\>>
  towards (hopefully) higher probability of data. (IMPLEMENTATION NOTE:
  concatenate all songs as a single time-serie and try to learn it).

  <strong|BB-RBM>

  Instead of GB-RBM which variance learning is complicated. I initially (and
  also) implement and test BB-RBM implementation as the RBM part of the
  RNN-RBM. In this case there is no variance terms <math|\<b-z\><rsub|i>> to
  worry about.

  <math|<frac|\<partial\>F|\<partial\>\<b-a\>>=-\<b-v\>>

  <math|<frac|\<partial\>F|\<partial\>\<b-b\>>=-sigmoid<around*|(|\<b-W\>*\<b-v\>+\<b-b\>|)>>

  <math|<frac|\<partial\>F|\<partial\>\<b-W\>>=-sigmoid<around*|(|\<b-W\>*\<b-v\>+\<b-b\>|)>\<b-v\><rsup|T>>

  \;

  NOTE: initial use of RNN-RBM outlined here seem to diverge quickly to chaos
  (many random notes played at once) when applying RNN-RBM to classical MIDI
  notes data (note range: C-4 .. B-6). It seems problem should be regularized
  somehow to limit number of on notes played at once.

  In practice it seems to be difficult to regularize RNN-RBM because of the
  special form of the error function (log probability). I suggest the use of
  \Pnegative gradient\Q. In addition to training samples which probability
  should be maximized, artificial songs are created where each note has
  random probability <math|p\<gtr\>0.50> of being in on position and gradient
  of these additional training samples is calculated normally but the
  calculated gradient is substracted from the positive gradient so that
  probability of those \Prandom songs\Q is greatly reduced.
</body>

<initial|<\collection>
</collection>>