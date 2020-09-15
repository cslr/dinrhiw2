<TeXmacs|1.99.12>

<style|generic>

<\body>
  <\padded-center>
    <\strong>
      Gradient of Complex Neural Network
    </strong>

    <em|tomas.ukkonen@novelinsight.fi>, 2020
  </padded-center>

  \;

  Complex valued neural networks extend computing capacity of neural
  networks. However, analytical calculation of the gradient of the network's
  error terms cannot be done using standard complex differentation and
  requires Wirtinger-calculus which defines function
  <math|\<b-f\><around*|(|\<b-z\>,<wide|\<b-z\>|\<bar\>>|)>> which is not
  fully holomorphic. Good reference paper about Wirtinger calculus is
  <with|font-shape|italic|The Complex Gradient Operator and the CR-Calculus.
  Ken Kreutz-Delgado. 2009. (arXiv.org 0906.4835v1)>.

  Consider a two-layer complex valued neural network

  <center|<math|\<b-y\><around*|(|\<b-z\>|)>=\<b-f\><around*|(|\<b-W\><rsup|<around*|(|2|)>>*\<b-g\><around*|(|\<b-W\><rsup|<around*|(|1|)>>\<b-z\>+\<b-b\><rsup|<around*|(|1|)>>|)>+\<b-b\><rsup|<around*|(|2|)>>|)>>.>

  We have linear algebra opetations <math|\<b-v\><rsup|<around*|(|1|)>>=\<b-W\><rsup|<around*|(|1|)>>\<b-z\>+\<b-b\><rsup|<around*|(|1|)>>>
  which satisfy Cauchy-Riemann conditions and which can be derivated using
  standard complex analysis. But additionally to this we have non-linearities
  which typically don't satisfy Cauchy-Riemann conditions. The standard
  non-linearities like leaky ReLU (rectifier
  <math|y=max<around*|(|0.01x,x|)>>) don't satisfy Cauchy-Riemann conditions.

  For simplicity we want to have non-linearity which satisfies Cauchy-Riemann
  conditions so that we can have analytical derivate of the function.

  Function <math|f<around*|(|z|)>=e<rsup|z>> satisfies Cauchy-Riemann
  conditions which can be seen by calculating partial derivates. Because we
  cannot use ReLU we can use <with|font-shape|italic|softplus> non-linearity
  which can be derivated using standard complex calculus.

  <\padded-center>
    <math|f<around*|(|z|)>=ln<around*|(|1+e<rsup|k*z>|)>/k,k=1.5>

    <math|f<rprime|'><around*|(|z|)>=<around*|(|1+e<rsup|-k*z>|)><rsup|-1>>
  </padded-center>

  After these definitions we can calculate gradient of the neural network
  <math|\<b-y\><around*|(|\<b-z\>|)>> function using simple standard complex
  calculus. But we need to additionally calculate derivate of error function\ 

  <\padded-center>
    <math|f<around*|(|\<b-w\>,<wide|\<b-w\>|\<bar\>>|)>=<frac|1|2>E<around*|{|\<b-z\><around*|(|\<b-w\>|)><rsup|H>\<b-z\><around*|(|\<b-w\>|)>|}>=<frac|1|2>E<around*|{|<around*|(|\<b-y\><around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>-\<b-y\>|)><rsup|H><around*|(|\<b-y\><around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>-\<b-y\>|)>|}>>.
  </padded-center>

  We can calculate partial derivates

  <\padded-center>
    <math|<frac|\<partial\>f|\<partial\>\<b-w\>>=<frac|1|2><around*|(|<frac|\<partial\><around*|\<\|\|\>|\<b-z\><around*|(|\<b-w\>|)>|\<\|\|\>><rsup|2>|\<partial\>*\<b-z\>>*<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>+<frac|\<partial\><frac|1|2><around*|\<\|\|\>|\<b-z\><around*|(|\<b-w\>|)>|\<\|\|\>><rsup|2>|\<partial\><wide|*\<b-z\>|\<bar\>>*><frac|\<partial\>*<wide|z|\<bar\>><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>|)>><math|>
  </padded-center>

  \ that <math|f<around*|(||)>> is real valued so we can calculate conjugate
  gradient by using the result: <math|<frac|\<partial\>f|\<partial\><wide|\<b-w\>|\<bar\>>>==<wide|<frac|\<partial\>f|\<partial\>*\<b-w\>>|\<bar\>>>.

  Additionally because neural network <math|\<b-z\><around*|(|\<b-w\>|)>=\<b-y\><around*|(|\<b-z\><around*|\||\<b-w\>|\<nobracket\>>|)>>
  satisfies Cauchy-Riemann conditions this means
  <math|<frac|\<partial\>*<wide|z|\<bar\>><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>=<around*|(|<wide|<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*<wide|\<b-w\>|\<bar\>>>|\<bar\>>|)>=\<b-0\>>.
  Also the partial derivate <math|<frac|\<partial\><around*|\<\|\|\>|\<b-z\><around*|(|\<b-w\>|)>|\<\|\|\>><rsup|2>|\<partial\>*\<b-z\>>=<frac|\<partial\><around*|(|\<b-z\><around*|(|\<b-w\>|)><rsup|H>\<b-z\><around*|(|\<b-w\>|)>|)>|\<partial\>*\<b-z\>>=\<b-z\><around*|(|\<b-w\>|)><rsup|H>>.
  This means we have derivates

  <\padded-center>
    <math|<frac|\<partial\>f<around*|(|\<b-w\>,<wide|\<b-w\>|\<bar\>>|)>|\<partial\>\<b-w\>>=<frac|1|2>\<b-z\><around*|(|\<b-w\>|)><rsup|H>*<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>><math|>

    <math|<frac|\<partial\>f<around*|(|\<b-w\>,<wide|\<b-w\>|\<bar\>>|)>|\<partial\><wide|\<b-w\>|\<bar\>>>==<wide|<frac|\<partial\>f|\<partial\>*\<b-w\>>|\<bar\>>>=<math|<frac|1|2>\<b-z\><around*|(|\<b-w\>|)><rsup|T>*<wide|<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>|\<bar\>>>
  </padded-center>

  Now we want to transform these to real valued coordinate system
  <math|r=<around*|(|\<b-x\><rsub|k>,y<rsub|k>|)>\<sim\>\<bbb-R\><rsup|2>>
  where real and imaginary values are separated.\ 

  This can be done using linear mapping <math|\<b-r\>=<frac|1|2>\<b-J\><rsup|H>\<b-c\>>,
  <math|\<b-c\>=<around*|[|<matrix|<tformat|<table|<row|<cell|\<b-z\>>>|<row|<cell|<wide|\<up-z\>|\<bar\>>>>>>>|]>>
  <math|>and <math|<frac|\<partial\>f|\<partial\>\<b-r\>>=\<b-J\><rsup|T><frac|\<partial\>f|\<partial\>\<b-c\>>>
  where <math|\<b-J\><rsup|T>=<matrix|<tformat|<table|<row|<cell|I>|<cell|I>>|<row|<cell|I*j>|<cell|-I*j>>>>>>
  so we get results (<math|\<b-w\>=\<b-x\>+\<b-y\>*j>):

  <\padded-center>
    <math|<frac|\<partial\>*Re<around*|(|f<around*|(|\<b-w\>|)>|)>|\<partial\>*\<b-x\>>=<frac|1|2><around*|(|\<b-z\><around*|(|\<b-w\>|)><rsup|H>*<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>+\<b-z\><around*|(|\<b-w\>|)><rsup|T>*<wide|<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>|\<bar\>>|)>>

    <math|<frac|\<partial\>*Im<around*|(|f<around*|(|\<b-w\>|)>|)>|\<partial\>*\<b-y\>>=<frac|1|2><around*|(|\<b-z\><around*|(|\<b-w\>|)><rsup|H>*<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>-\<b-z\><around*|(|\<b-w\>|)><rsup|T>*<wide|<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>|\<bar\>>|)>j>
  </padded-center>

  Plugging these equations into formula <math|f<around*|(|\<b-w\>|)>=\<b-x\>+\<b-y\>*j=Re<around*|(|f<around*|(|\<b-w\>|)>|)>+Im<around*|(|f<around*|(|\<b-w\>|)>|)>j>
  gives

  <\padded-center>
    <math|<frac|\<partial\>*\<b-f\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>=<frac|1|2><around*|(|\<b-z\><around*|(|\<b-w\>|)><rsup|H>*<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>+\<b-z\><around*|(|\<b-w\>|)><rsup|T>*<wide|<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>|\<bar\>>|)>-<frac|1|2><around*|(|\<b-z\><around*|(|\<b-w\>|)><rsup|H>*<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>-\<b-z\><around*|(|\<b-w\>|)><rsup|T>*<wide|<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>|\<bar\>>|)>>

    <math|<frac|\<partial\>*\<b-f\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>=\<b-z\><around*|(|\<b-w\>|)><rsup|T>*<wide|*<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>|\<bar\>>>
  </padded-center>

  In practice optimization seem to generate large weight values
  <math|\<b-w\>> which cause floating point errors. This means we need to add
  regularizer to our optimized function.

  <\padded-center>
    <math|f<around*|(|\<b-w\>,<wide|\<b-w\>|\<bar\>>|)>=<frac|1|2>E<around*|{|\<b-z\><around*|(|\<b-w\>|)><rsup|H>\<b-z\><around*|(|\<b-w\>|)>|}>+<frac|1|2><around*|\<\|\|\>|\<b-w\>|\<\|\|\>><rsup|2>=<frac|1|2>E<around*|{|<around*|(|\<b-y\><around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>-\<b-y\>|)><rsup|H><around*|(|\<b-y\><around*|(|\<b-x\><around*|\||\<b-w\>|\<nobracket\>>|)>-\<b-y\>|)>|}>+<frac|1|2>\<b-alpha\>*<around*|\<\|\|\>|\<b-w\>|\<\|\|\>><rsup|2>>.
  </padded-center>

  This means\ 

  <\padded-center>
    <math|<frac|\<partial\>f<around*|(|\<b-w\>,<wide|\<b-w\>|\<bar\>>|)>|\<partial\>\<b-w\>>=<frac|1|2>\<b-z\><around*|(|\<b-w\>|)><rsup|H>*<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>><math|+<frac|1|2>*\<b-alpha\>**<wide|\<b-w\>|\<bar\>>>

    <math|<frac|\<partial\>f<around*|(|\<b-w\>,<wide|\<b-w\>|\<bar\>>|)>|\<partial\><wide|\<b-w\>|\<bar\>>>==<wide|<frac|\<partial\>f|\<partial\>*\<b-w\>>|\<bar\>>>=<math|<frac|1|2>\<b-z\><around*|(|\<b-w\>|)><rsup|T>*<wide|<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>|\<bar\>>+<frac|1|2>*\<b-alpha\>*\<b-w\>>
  </padded-center>

  And real/imaginary parts are:

  <\padded-center>
    <math|<frac|\<partial\>*Re<around*|(|f<around*|(|\<b-w\>|)>|)>|\<partial\>*\<b-x\>>=<frac|1|2><around*|(|\<b-z\><around*|(|\<b-w\>|)><rsup|H>*<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>+\<b-z\><around*|(|\<b-w\>|)><rsup|T>*<wide|<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>|\<bar\>>|)>+\<b-alpha\>*Re<around*|(|\<b-w\>|)>>

    <math|<frac|\<partial\>*Im<around*|(|f<around*|(|\<b-w\>|)>|)>|\<partial\>*\<b-y\>>=<frac|1|2><around*|(|\<b-z\><around*|(|\<b-w\>|)><rsup|H>*<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>-\<b-z\><around*|(|\<b-w\>|)><rsup|T>*<wide|<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>|\<bar\>>|)>j+\<b-alpha\>*Im<around*|(|\<b-w\>|)>>
  </padded-center>

  <\padded-center>
    <math|<frac|\<partial\>*\<b-f\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>=\<b-z\><around*|(|\<b-w\>|)><rsup|T>*<wide|*<frac|\<partial\>*\<b-z\><around*|(|\<b-w\>|)>|\<partial\>*\<b-w\>>|\<bar\>>+\<b-alpha\>*\<b-w\>>
  </padded-center>

  Adding regularizer term with <math|\<alpha\>=0.01> solves the problem
  (weight vectors are initially <math|N<around*|(|0,I|)>/sqrt<around*|(|dim<around*|(|w|)>|)>>
  or something so <math|N<around*|(|0,I|)>> input data should initially have
  something like unit variance and zero mean when going through the linear
  network.

  <em|Although this neural network model uses complex numbers it fulfills
  Cauchy-Riemann conditions so it restricts the possible complex number
  functions. Improvement would use non standard complex derivable
  nonlinearities like extension of LeakyRELU extended complex numbers. After
  testing quality of found solutions test how TensorFlow's implementation of
  complex numbers with non-standard functions (automatic derivation)
  performs. >

  <with|font-series|bold|TODO: Calculate also gradient of norm
  <math|<around*|\<\|\|\>|\<b-f\><around*|(|\<b-z\>|)>-\<b-y\>|\<\|\|\>>>
  instead of squared error>.

  \ 
</body>

<initial|<\collection>
</collection>>