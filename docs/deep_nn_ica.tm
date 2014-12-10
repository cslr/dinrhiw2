<TeXmacs|1.0.7.15>

<style|generic>

<\body>
  <\doc-data|<doc-title|Neural network independent component analysis
  pre-training algorithm>|<doc-author-data|<author-name|Tomas Ukkonen,
  tomas.ukonen@iki.fi, 2014>>>
    \;

    \;
  </doc-data|>

  One approach in ICA is to try to diagonalize non-linear matrix
  <math|\<b-G\><rsub|\<b-x\>x>=E<around*|{|g<around*|(|\<b-x\>|)>g<around*|(|\<b-x\>|)><rsup|H>|}>>
  where <math|g<around*|(|x|)>> function has been shown to create interesting
  higher order moments. Now assuming <math|E<around*|{|g<around*|(|x|)>|}>=0>,
  diagonalization of the matrix means that
  <math|E<around*|{|g<around*|(|x<rsub|i>|)>*g<around*|(|x<rsub|j>|)>|}>=E<around*|{|g<around*|(|x<rsub|i>|}>*E<around*|{||\<nobracket\>>g<around*|(|x<rsub|j>|)>|}>>,
  one consequence of the independence of the <math|x<rsub|i>> and
  <math|x<rsub|j>>. Additionally, it is possible to define function
  <math|g<around*|(|x|)>=g<around*|(|x|)>-E<around*|{|g<around*|(|x|)>|}>>
  which always satisfies the condition <math|E<around*|{|g<around*|(|x|)>|}>=0>.

  Now what is good non-linearity for <math|g<around*|(|x|)>>? It seems that
  <math|g<around*|(|x|)>=sinh<around*|(|x|)>> is rather good giving that it
  is zero preserving, odd function (as long as we don't add the
  <math|E<around*|{|g<around*|(|x|)>|}>> term). And it's Taylor's expansion
  gives only positive coefficients for <math|x<rsup|2k+1>> terms meaning that
  it should ``push' positive and negative moments into different directions.

  How to diagonalize <math|\<b-G\><rsub|\<b-x\>x>> then? The solution to this
  question is almost trivial. <math|\<b-y\>=g<rsup|-1><around*|(|\<b-W\>\<b-x\>|)>>
  diagonalizes <math|\<b-G\><rsub|\<b-x\>x>> matrix when <math|\<b-W\>>
  diagonalizes the <math|E<around*|{|\<b-x\>\<b-x\><rsup|H>|}>> matrix. That
  is, it is solution to linear ICA (or PCA). Additionally, it is interesting
  that inverse of <math|sinh<around*|(|x|)>> is <math|asinh<around*|(|x|)>> a
  function that is close to <math|tanh<around*|(|x|)>> or sigmoidal functions
  that have been proposed as non-linearities for neural networks.

  The whole algorithm for pre-training neural networks is then to stimulate
  neural network, collect input samples for each layer and sequentially
  calculate ICA solution for the inputs and use <math|asinh<around*|(|x|)>>
  non-linearity instead of more standard ones.

  <\with|par-mode|center>
    <math|\<b-x\><rsub|n+1>=g<rsup|-1><around*|(|\<b-W\>*\<b-x\><rsub|n>|)>=asinh<around*|(|\<b-W\>*\<b-x\><rsub|n>+E|{>sinh<around*|(|\<b-x\><rsub|n>|)><around*|}||)>>
  </with>

  Here we have assumed <math|E<around*|{|\<b-x\><rsub|n>|}>=0> as can be done
  as a preprocessing step for the first layer, but it is easy to add
  additional bias term <math|-\<b-W\>*E<around*|{|\<b-x\><rsub|n>|}>> to the
  formula to correct for non-zero mean values. The beauty of the solution
  comes from that it fits to the previous neural network structures and
  jointly diagnonalizes <math|E<around*|{|\<b-x\>*\<b-x\><rsup|H>|}>> and
  <math|E<around*|{|g<around*|(|\<b-x\>|)>g<around*|(|\<b-x\>|)><rsup|H>|}>>
  matrixes and tells how to integrate ICA solutions into non-linear neural
  networks.

  After the pretraining step to initialize neural network weight and bias
  terms. Traditional optimization methods can be then used to solve for the
  problem <math|\<b-y\>=f<around*|(|\<b-x\>|)>> where out network now a a
  default extracts non-linear independent components from the data.

  NOTE: PCA/ICA solutions should have variance of 4-6 or something as
  <math|asinh<around*|(|x|)>> non-linearity operates with much larger values
  than <math|tanh<around*|(|x|)>> or other functions.

  <strong|Further work>

  Test how this actually works with different <math|g<around*|(|x|)>>
  non-linearities.

  \;
</body>