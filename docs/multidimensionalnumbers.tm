<TeXmacs|1.99.12>

<style|<tuple|generic|old-spacing|old-dots>>

<\body>
  <with|font-series|bold|Increasing Calculation Capacity By Number Theoretic
  Extension>

  \;

  <with|font-series|bold|1. Multidimensional numbers>

  <with|font-shape|italic|\P..to divide divine.\Q>

  Tomas Ukkonen, 2015

  \;

  We extend our real number system <math|\<bbb-R\>> by having dimensional
  basis. Instead of pure scalar <math|\<alpha\>>, we define dimensional
  numbers <math|\<alpha\>*r<rsup|d>> with dimension <math|d> which can be any
  real number (including negative numbers) but which we restrict here to be
  integer number.\ 

  <with|font-series|bold|Motivation>

  This approach can be motivated by trying to measure dimensionality of
  Koch's snowflake fractal (with real numbered dimension).

  If we look at one side of the fractal we can see that it is self-repeating,
  one \Pmeasure stick length\Q of fractal can be used four times to measure
  one side of the snowflake, but equal result can be got by \Pstretching
  fractal measure stick\Q to be 3 times larger (longer). This leads into
  equation\ 

  <center|<math|4*r<rsup|d>=<around*|(|3*r|)><rsup|d>>>

  And solving this for <math|d=log<around*|(|4|)>/log<around*|(|3|)>=1.2619>
  leads into real numbered dimension higher than one. It is then no wonder
  that length of snowflake (or fractal's) curve is infinite. The dimension is
  larger than one and we are measuring with \P1d measure stick\Q
  <math|\<alpha\>*r<rsup|1.2619>/r<rsup|1>=\<alpha\>*r<rsup|0.2619>> and our
  measurement in dimension 1 is infinite (<math|r<rsup|\<beta\>>=\<infty\>>
  when <math|\<beta\>\<neq\>0>).

  <with|font-shape|italic|Another example: integration through sum.>

  Another idea: black holes in physics are generated when mass increases to
  \Pinfinity\Q. In our theory this means that dimension of the system
  increases (like 3d fractal). In higher dimensions <math|<around*|(|7+|)>>
  surface area of a unit hypersphere becomes smaller and smaller meaning that
  more and more volume can fit to the same surface area (most of the volume
  is near edges and volume of center is insignificantly small). This then
  means that things are more and more separated from each other (all states
  are different). On the other extreme, in a perfect vacuum (quantum
  physics), mass goes to zero. This could then be intepreted as \Psubzero\Q
  values meaning that dimension of the objects become smaller than <math|1>.
  This means that surface area of sub 1-dimensional \Pfractal\Q unit sphere
  (proof?) becomes smaller and smaller with same volume and more importantly
  things are more and more concentrated to the same point meaning that things
  and states are inseparable.

  <with|font-series|bold|Mathematics (discrete computer implementation)>

  We can define our multidimensional numbers (integer dimensions) as a vector
  sum <math|a=<big|sum><rsup|D-1><rsub|i=0>\<alpha\><rsub|i>r<rsup|i>>. Here
  we can see this is not only \Ppolynomial\Q but also vector space because
  dimensions <math|r<rsup|i>> are perpendicual to each other, there is no
  real number <math|\<alpha\>> which can be used get into another dimension
  (only way to get into another dimension through division by zero or by
  process which leads to infinity: <math|1/\<alpha\>,\<alpha\>\<rightarrow\>0>
  or <math|\<alpha\>,\<alpha\>\<rightarrow\>\<infty\>>). The way to get from
  higher dimensions into lower dimensions is through division by infinity:
  <math|1/\<alpha\>,\<alpha\>\<rightarrow\>\<infty\>>. Additionally, the
  number of dimensions is restricted to a <with|font-series|bold|prime
  number> <math|P> in order to create closed system where calculation of
  inverse is always possible. We now define addition and multiplication as
  follows:

  <\center>
    <math|a+b=<big|sum><rsup|P><rsub|i=0><around*|(|\<alpha\><rsub|i>+\<beta\><rsub|i>|)>r<rsup|i>>

    <math|a*b=<big|sum><rsub|i,j>\<alpha\><rsub|i>*\<beta\><rsub|j>*r<rsup|<around*|(|i+j|)>
    mod P>>
  </center>

  By resticting exponents to a modular arithmetic we can create a number
  system which is well defined and always works. This also means that
  \Pdivision by zero\Q is always mathematically well defined and all numbers
  (including \Pzero\Q) has always somekind of inverse although by restricting
  the number of dimensions to <math|P> means that higher dimensions can
  \Poverflow\Q back to lower dimensions and results into unintuitive results.

  Required: proof that number system defined this way works. (requires
  Fermat's theorem and discrete mathematics). I did this earlier but lost
  documents. Partial mini proof: multiplication of
  <math|r<rsup|i>*r<rsup|j>=r<rsup|<around*|(|i+j|)> mod P>> leads to number
  system which have always inverse and is well defined (normal modular
  arithmetic). Therefore multiplication of <math|r<rsup|d>>:s are well
  defined. The right name for this kind of numbers are <em|<strong|polynomial
  rings>> which are well-studied in discrete mathematics.

  <with|font-series|bold|Algorithm for division>

  Next we want algorithm to calculate <math|a<rsup|-1>> in order to divide
  <math|a/b> (even by zero). We notice that the circular structure used in
  multiplication is similar to discrete finite Fourier transform which also
  uses circular buffers/circular convolution. Therefore somekind of transform
  approach could maybe make sense.

  <with|font-series|bold|TODO: implement code which creates working
  multidimensional numbers>

  <verbatim|dinrhiw2/math/modular.h> has integer modular integer code which
  can be used to implement superdimensional numbers. Modify
  <verbatim|dinrhiw2/math/superrresolution.h> to work with modular integer
  math of polynomial exponents. Implement circular convolution and solve
  multiplication and inverse using Fourier transform and inverse Fourier
  transform (KissFFT has BSD style license).

  Write testcase to check multidimensional number code work always (real and
  complex number coefficients) and then use it as datatype for
  <verbatim|nnetwork\<less\>\<gtr\>>.

  \;

  <with|font-series|bold|2. Extending Neural Network Capacity>

  After showing multidimensional numbers fulfills number theoretic properties
  of <with|font-series|bold|field/ring(?)>. We can use it as our neural
  networks numbers and test the improved computing capacities of
  multidimensional neural network.

  - how to calculate derive and calculate gradient of multidimensional
  numbers(?). [modular arithmetic loses its capacity to compare largeness of
  numbers so in general we cannot say functions are continuous in a typical
  sense. We can define gradient in each dimension <math|d> but what are
  requirements for general differentiability(?). In complex analysis this is
  already a problem.

  - initially implement random search and compare its results to real number
  version. Multiplications are now very non-linear (\Pcryptographic\Q
  non-linearities) so learning functions should be complicated and gradient
  descent may not work very well(??).

  \ \ 

  \;

  \;
</body>

<initial|<\collection>
</collection>>