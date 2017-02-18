<TeXmacs|1.0.7.15>

<style|generic>

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

  The gradients of the final layer are (non-zero terms are at the <math|j>:th
  row):

  <\center>
    <math|<frac|\<partial\>*y<around*|(|\<b-x\>|)>|\<partial\>*w<rsub|j*i><rsup|<around*|(|2|)>>>=diag<around*|(|<frac|\<partial\>f<around*|(|\<b-x\>|)>|\<partial\>\<b-x\>>|)>*<matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|g<rsub|i>>>|<row|<cell|0>>>>>>

    <math|<frac|\<partial\>*y<around*|(|\<b-x\>|)>|\<partial\>*b<rsub|j><rsup|<around*|(|2|)>>>=diag<around*|(|<frac|\<partial\>f<around*|(|\<b-x\>|)>|\<partial\>\<b-x\>>|)>*<matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|1>>|<row|<cell|0>>>>>>
  </center>

  The derivation chain-rule can be used to calculate the second (and more
  deep layers' gradients):

  <\center>
    <math|<frac|\<partial\>*y<around*|(|\<b-x\>|)>|\<partial\>*w<rsub|j*i><rsup|<around*|(|1|)>>>=diag<around*|(|<frac|\<partial\>f<around*|(|\<b-x\>|)>|\<partial\><around*|(|\<b-W\><rsup|<around*|(|2|)>>\<b-g\>+\<b-b\><rsup|<around*|(|2|)>>|)>>|)>*<frac|\<partial\><around*|(|\<b-W\><rsup|<around*|(|2|)>>*\<b-g\>+\<b-b\><rsup|<around*|(|2|)>>|)>|\<partial\>\<b-g\>>*<frac|\<partial\>\<b-g\><around*|(|\<b-x\>|)>|\<partial\><around*|(|\<b-W\><rsup|<around*|(|1|)>>\<b-x\>+\<b-b\><rsup|<around*|(|1|)>>|)>>*<frac|\<partial\><around*|(|\<b-W\><rsup|<around*|(|2|)>>\<b-x\>+\<b-b\><rsup|<around*|(|2|)>>|)>|\<partial\>*w<rsup|2><rsub|j*i>>>

    <math|<frac|\<partial\>*y<around*|(|\<b-x\>|)>|\<partial\>*w<rsub|j*i><rsup|<around*|(|1|)>>>=diag<around*|(|<frac|\<partial\>f<around*|(|\<b-x\>|)>|\<partial\><around*|(|\<b-W\><rsup|<around*|(|2|)>>\<b-g\>+\<b-b\><rsup|<around*|(|2|)>>|)>>|)>*\<b-W\><rsup|<around*|(|2|)>>*diag<around*|(|*<frac|\<partial\>\<b-g\><around*|(|\<b-x\>|)>|\<partial\><around*|(|\<b-W\><rsup|<around*|(|1|)>>\<b-x\>+\<b-b\><rsup|<around*|(|1|)>>|)>>|)>*<matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|x<rsub|i>>>|<row|<cell|0>>>>>>

    <math|<frac|\<partial\>*y<around*|(|\<b-x\>|)>|\<partial\>*b<rsup|<around*|(|1|)>><rsub|j>>=diag<around*|(|<frac|\<partial\>f<around*|(|\<b-x\>|)>|\<partial\><around*|(|\<b-W\><rsup|<around*|(|2|)>>\<b-g\>+\<b-b\><rsup|<around*|(|2|)>>|)>>|)>*\<b-W\><rsup|<around*|(|2|)>>*diag<around*|(|*<frac|\<partial\>\<b-g\><around*|(|\<b-x\>|)>|\<partial\><around*|(|\<b-W\><rsup|<around*|(|1|)>>\<b-x\>+\<b-b\><rsup|<around*|(|1|)>>|)>>|)>*<matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|1>>|<row|<cell|0>>>>>>
  </center>

  By analysing the chain rule we can derive generic backpropagation formula
  for the full gradient. Let <math|\<b-v\><rsup|<around*|(|k|)>>> be a
  <math|k>:th layers local field, <math|\<b-v\><rsup|<around*|(|k|)>>=\<b-W\><rsup|<around*|(|k|)>>f<around*|(|\<b-v\><rsup|<around*|(|k-1|)>>|)>+\<b-b\><rsup|<around*|(|k|)>>>.
  Then local gradient matrices <math|\<b-delta\><rsup|<around*|(|k|)>>> are

  <\center>
    <math|\<b-delta\><rsup|<around*|(|L|)>>=diag<around*|(|<frac|\<partial\>\<b-f\><around*|(|\<b-v\><rsup|<around*|(|L|)>>|)>|\<partial\>\<b-v\><rsup|<around*|(|L|)>>>|)>>

    <math|\<b-delta\><rsup|<around*|(|k-1|)>>=\<b-delta\><rsup|<around*|(|k|)>>*\<b-W\><rsup|<around*|(|k|)>>*diag<around*|(|<frac|\<partial\>\<b-f\><around*|(|\<b-v\><rsup|<around*|(|k-1|)>>|)>|\<partial\>\<b-v\><rsup|<around*|(|k-1|)>>>|)>>
  </center>

  And network's parameter gradient matrices for each layer are (only
  <math|j>:th element of each row is non-zero):\ 

  <\center>
    <math|<frac|\<partial\>*y<around*|(|\<b-x\>|)>|\<partial\>*w<rsub|j*i><rsup|<around*|(|k|)>>>=\<b-delta\><rsup|<around*|(|k|)>>*<matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|f<around*|(|v<rsub|i><rsup|<around*|(|k-1|)>>|)>>>|<row|<cell|0>>>>>>

    <math|<frac|\<partial\>*y<around*|(|\<b-x\>|)>|\<partial\>*b<rsub|j><rsup|<around*|(|k|)>>>=\<b-delta\><rsup|<around*|(|k|)>><matrix|<tformat|<table|<row|<cell|0>>|<row|<cell|1>>|<row|<cell|0>>>>>>
  </center>

  To test that gradient matrix is correctly computed it can be compared with
  normal squared error calculations (normal backpropagation).

  <center|<math|\<varepsilon\><around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>=<frac|1|2><around*|\<\|\|\>|y<rsub|i>-y<around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>|\<\|\|\>><rsup|2>>>

  <center|<math|<frac|\<partial\>\<varepsilon\><around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>|\<partial\>\<b-w\>>=<around*|(|y<around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>-y<rsub|i>|)><rsup|T>*<frac|\<partial\>y<around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>|\<partial\>*\<b-w\>>>>

  \;

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

  <strong|Recurrent Neural Networks and Backpropagation Through Time (BPTT)>

  The basic learning algorithm for recurrent neural networks (RNN) is
  <with|font-series|bold|backpropagation through time>
  (<with|font-shape|italic|BPTT>). This is done by unfolding neural net in
  time an computing the gradients. The recurrent neural network is

  <center|<math|\<b-z\><around*|(|n+1|)>=<matrix|<tformat|<table|<row|<cell|\<b-y\><around*|(|n+1|)>>>|<row|<cell|\<b-r\><around*|(|n+1|)>>>>>>=\<b-f\><around*|(|\<b-x\><around*|(|n|)>,\<b-r\><around*|(|n|)>|)>>>

  The error function to minimize is:

  <center|<math|E<around*|(|N|)>=<frac|1|2><big|sum><rsup|N><rsub|n=0><around*|\<\|\|\>|\<b-d\><around*|(|n+1|)>-\<b-Gamma\><rsub|\<b-y\>>\<b-f\><around*|(|\<b-x\><around*|(|n+1|)>,\<b-Gamma\><rsub|\<b-r\>>\<b-z\><around*|(|n|)><around*|\||\<b-w\>|\<nobracket\>>|)>|\<\|\|\>><rsup|2>>>

  In which <math|\<b-Gamma\>> matrices are used to select
  <math|\<b-y\><around*|(|n|)>> and <math|\<b-r\><around*|(|n|)>> vectors
  from generic output vector and the initial input to feedforward neural
  network is zero <math|\<b-z\><around*|(|0|)>=\<b-0\>>.

  It is possible to calculate gradient of <math|\<b-f\>> using the chain rule

  <center|<math|<frac|\<partial\>E<around*|(|N|)>|\<partial\>\<b-w\>>=<big|sum><rsup|N><rsub|n=0><around*|(|\<b-Gamma\><rsub|\<b-y\>>\<b-f\><around*|(|\<b-x\><around*|(|n+1|)>,\<b-Gamma\><rsub|\<b-r\>>\<b-z\><around*|(|n|)><around*|\||\<b-w\>|\<nobracket\>>|)>-\<b-d\><around*|(|n+1|)>|)><rsup|T>*\<b-Gamma\><rsub|\<b-y\>>*\<nabla\><rsub|\<b-w\>>\<b-f\><around*|(|\<b-x\><around*|(|n+1|)>,\<b-Gamma\><rsub|\<b-r\>>\<b-z\><around*|(|n|)><around*|\||\<b-w\>|\<nobracket\>>|)>>>

  To calculate the gradient <math|\<nabla\><rsub|\<b-w\>>\<b-f\>> one must
  remember that <math|\<b-z\><around*|(|n|)>> now also depends on
  <math|\<b-w\>> resulting into eq:

  <center|<math|\<nabla\><rsub|\<b-w\>>\<b-f\><around*|(|\<b-x\><around*|(|n+1|)>,\<b-Gamma\><rsub|\<b-r\>>\<b-z\><around*|(|n|)>|)>=<frac|\<partial\>\<b-f\>|\<partial\>\<b-w\>>+<frac|\<partial\>\<b-f\>|\<partial\>\<b-r\>>*\<b-Gamma\><rsub|\<b-r\>><frac|\<partial\>\<b-z\><around*|(|n|)>|\<partial\>\<b-w\>>*<rsub|>>>

  To further compute gradients we get a generic update rule

  <center|<math|<frac|\<partial\>*\<b-z\><around*|(|n|)>|\<partial\>\<b-w\>>=<frac|\<partial\>\<b-f\>|\<partial\>\<b-w\>>+<frac|\<partial\>\<b-f\>|\<partial\>\<b-r\>>*\<b-Gamma\><rsub|\<b-r\>>*<frac|\<partial\>\<b-z\><around*|(|n-1|)>|\<partial\>\<b-w\>>>>

  The computation of gradients can be therefore bootstrapped by setting
  <math|<frac|\<partial\>\<b-z\><around*|(|1|)>|\<partial\>\<b-w\>>=<frac|\<partial\>\<b-f\><around*|(|\<b-0\>,\<b-0\>|)>|\<partial\>\<b-w\>>>
  and iteratively updating <math|\<b-z\>> gradient while computing the
  current error for the timestep.

  \;

  \;
</body>