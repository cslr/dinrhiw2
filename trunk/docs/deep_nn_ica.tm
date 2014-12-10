<TeXmacs|1.0.7.15>

<style|generic>

<\body>
  <\doc-data|<doc-title|Neural network principal component analysis
  pre-training algorithm>|<doc-author-data|<author-name|Tomas Ukkonen,
  tomas.ukonen@iki.fi, 2014>>>
    \;

    \;
  </doc-data|>

  One approach in ICA is to try to diagonalize non-linear matrix
  <math|\<b-G\><rsub|\<b-x\>\<b-x\>>=E<around*|{|g<around*|(|\<b-x\>|)>g<around*|(|\<b-x\>|)><rsup|H>|}>>
  where <math|g<around*|(|x|)>> function has been shown to create interesting
  higher order moments. Now assuming <math|E<around*|{|g<around*|(|\<b-x\>|)>|}>=0\<nocomma\>>,
  diagonalization of the matrix means that
  <math|E<around*|{|g<around*|(|x<rsub|i>|)>*g<around*|(|x<rsub|j>|)>|}>=E<around*|{|g<around*|(|x<rsub|i>|}>*E<around*|{||\<nobracket\>>g<around*|(|x<rsub|j>|)>|}>>,
  one consequence of the independence of the <math|x<rsub|i>> and
  <math|x<rsub|j>>. Additionally, it is possible to define function
  <math|g<rprime|'><around*|(|x|)>=g<around*|(|x|)>-E<around*|{|g<around*|(|x|)>|}>>
  which always satisfies the condition <math|E<around*|{|g<rprime|'><around*|(|x|)>|}>=0>.

  Now what is good non-linearity for <math|g<around*|(|x|)>>? It seems that
  <math|g<around*|(|x|)>=sinh<around*|(|x|)>> is rather good giving that it
  is zero preserving, odd function (as long as we don't add the
  <math|E<around*|{|g<around*|(|x|)>|}>> term). And it's Taylor's expansion
  gives only positive coefficients for <math|x<rsup|2k+1>> terms meaning that
  it should ``push' positive and negative moments into different directions.

  How to diagonalize <math|\<b-G\><rsub|\<b-x\>x>> then? The solution to this
  question is almost trivial. Solution <math|\<b-y\>=g<rprime|'><rsup|-1><around*|(|\<b-W\><around*|(|\<b-x\>-E<around*|{|\<b-x\>|}>|)>|)>>
  diagonalizes <math|\<b-G\><rsub|\<b-x\>\<b-x\>>> matrix when <math|\<b-W\>>
  diagonalizes the <math|E<around*|{|<around*|(|\<b-x\>-E<around*|{|\<b-x\>|}>|)><around*|(|\<b-x\>-E<around*|{|\<b-x\>|}>|)><rsup|H>|}>>
  matrix. That is, it is solution to linear ICA (or PCA). Additionally, it is
  interesting that inverse of <math|sinh<around*|(|x|)>> is
  <math|asinh<around*|(|x|)>> a function that is close to
  <math|tanh<around*|(|x|)>> or sigmoidal functions that have been proposed
  as non-linearities for neural networks.

  <\with|par-mode|center>
    <math|g<rprime|'><rsup|-1><around*|(|\<b-x\>|)>=g<rsup|-1><around*|(|\<b-x\>+E<around*|{|g<around*|(|\<b-x\>|)>|}>|)>>
  </with>

  The whole algorithm for pre-training neural networks is then to stimulate
  neural network, collect input samples for each layer and sequentially
  calculate ICA solution for the inputs and use <math|asinh<around*|(|x|)>>
  non-linearity instead of more standard ones. So we have:\ 

  <\with|par-mode|center>
    <math|g<around*|(|\<b-x\><rsub|n+1>|)>-E<around*|{|g<around*|(|\<b-x\><rsub|n+1>|)>|}>=\<b-W\><around*|(|\<b-x\><rsub|n>-E<around*|{|\<b-x\><rsub|n>|}>|)>>

    <math|\<b-x\><rsub|n+1>=g<rsup|-1><around*|(|\<b-W\><around*|(|\<b-x\><rsub|n>-E<around*|{|\<b-x\><rsub|n>|}>|)>+E<around*|{|g<around*|(|\<b-x\><rsub|n+1>|)>|}>|)>>
  </with>

  Now we have to solve equation of the formula

  <\with|par-mode|center>
    <math|\<b-x\><rsub|n+1>=f<around*|(|E<around*|{|\<b-x\><rsub|n+1>|}>|)>>.
  </with>

  But if we notice that if we set <math|E<around*|{|g<around*|(|\<b-x\><rsub|n+1>|)>|}>=\<b-y\>>,
  then insert <math|\<b-x\><rsub|n+1>=g<rsup|-1><around*|(|\<b-W\><around*|(|\<b-x\><rsub|n>-E<around*|{|\<b-x\><rsub|n>|}>|)>+E<around*|{|g<around*|(|\<b-x\><rsub|n+1>|)>|}>|)>>
  and from it follows <math|E<around*|{|g<around*|(|\<b-x\><rsub|n+1>|)>|}>=E<around*|{|\<b-W\><around*|(|\<b-x\><rsub|n>-\<b-E\><around*|{|\<b-x\><rsub|n>|}>|)>+\<b-y\>|}>=\<b-W\>*E<around*|{|\<b-x\><rsub|n>-E<around*|{|\<b-x\><rsub|n>|}>|}>+\<b-y\>=\<b-y\>>.
  So <math|E<around*|{|sinh<around*|(|\<b-x\><rsub|n+1>|)>|}>=\<b-y\>> is
  solution to the equation and we can pick any <math|\<b-y\>> and choose
  <math|\<b-y\>=0>, and we get

  <\with|par-mode|center>
    <math|\<b-x\><rsub|n+1>=g<rsup|-1><around*|(|\<b-W\><around*|(|\<b-x\><rsub|n>-E<around*|{|\<b-x\><rsub|n>|}>|)>|)>>

    <math|\<b-x\><rsub|n+1>=asinh<around*|(|\<b-W\>*<around*|(|\<b-x\><rsub|n>-E<around*|{|\<b-x\><rsub|n>|}>|)>|)>>.
  </with>

  Note that the decision to pick <math|E<around*|{|g<around*|(|\<b-x\><rsub|n+1>|)>|}>>
  value is not maybe straightforward. But if <math|g<around*|(|x|)>> is odd
  function and the distribution has zero mean. Then for a odd function
  <math|E<around*|{|g<around*|(|\<b-x\>|)>|}>=0> can be a good solution.
  Another, <strong|better >solution can be
  <math|E<around*|{|g<around*|(|\<b-x\>|)>|}>=E<around*|{|\<b-x\>|}>> meaning
  that <math|g<around*|(|\<b-x\>|)>> is now statistically speaking <em|linear
  function>, and so should be also its inverse
  <math|E<around*|{|g<rsup|-1><around*|(|\<b-x\>|)>|}>\<thickapprox\>\<b-x\>>.
  This then leads into equation:

  <\with|par-mode|center>
    <math|\<b-x\><rsub|n+1>\<thickapprox\>E<around*|{|\<b-W\><around*|(|\<b-x\><rsub|n>-E<around*|{|\<b-x\><rsub|n>|}>|)>|}>+E<around*|{|\<b-x\><rsub|n+1>|}>>
  </with>

  which is of course <em|linear ICA solution> (so we get a ``non-linear''
  solution that are close to the linear solutions. Or, selecting
  <math|E<around*|{|g<around*|(|\<b-x\>|)>|}>=E<around*|{|\<b-x\>|}>> can
  also mean that the underlying distribution is modified to be
  <math|p<around*|(|x|)>\<sim\><frac|x|g<around*|(|x|)>>p<around*|(|x|)>=<frac|x|sinh<around*|(|x|)>>p<around*|(|x|)>>.
  And now, <math|<frac|x|sinh<around*|(|x|)>>> has a interesting, gaussian,
  bell function like shape meaning that distribution is modified to be
  concentrated around zero with a quick\ 

  Another way to see it, is to <with|font-shape|italic|choose non-linearity>
  so that <math|E<around*|{|g<around*|(|\<b-x\>|)>|}>\<thickapprox\>\<b-0\>>
  based on the distribution of data in the network.

  <\with|par-mode|center>
    \ <math|<big|int>g<around*|(|x<rsub|i>|)>p<around*|(|x<rsub|i>|)>d*x<rsub|i>=0>
  </with>

  The beauty of the solution comes from that it fits to the previous neural
  network structures and jointly diagnonalizes
  <math|E<around*|{|\<b-x\>*\<b-x\><rsup|H>|}>> and
  <math|E<around*|{|g<around*|(|\<b-x\>|)>g<around*|(|\<b-x\>|)><rsup|H>|}>>
  matrixes and tells how to integrate ICA solutions into non-linear neural
  networks.

  After the pretraining step to initialize neural network weight and bias
  terms. Traditional optimization methods can be then used to solve for the
  problem <math|\<b-y\>=f<around*|(|\<b-x\>|)>> where out network now a a
  default extracts non-linear independent components from the data.

  <em|NOTE: PCA/ICA solutions should have variance of 2.0 or something as
  <math|asinh<around*|(|x|)>> has different scaling than, for example,
  sinh(x)>.

  <strong|Further work>

  Test how this actually works with different <math|g<around*|(|x|)>>
  non-linearities.

  NOTE: because <math|g<around*|(|x|)>=tanh<around*|(|x|)>> is another useful
  non-linearity when calculating ICA. It might make sense to try to use
  <math|atanh<around*|(|x|)>> non-linearity for neural networks. DOES NOT
  WORK

  \;
</body>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
  </collection>
</references>