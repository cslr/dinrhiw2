<TeXmacs|1.0.7.15>

<style|generic>

<\body>
  <\doc-data|<doc-title|Neural network principal component analysis
  pre-training algorithm>|<doc-author-data|<author-name|Tomas Ukkonen,
  tomas.ukonen@iki.fi, 2014>>>
    \;

    \;
  </doc-data|>

  One approach in ICA is to try to diagonalize non-linear matrix\ 

  <\with|par-mode|center>
    <math|\<b-G\><rsub|\<b-x\>\<b-x\>>=E<around*|{|<around*|(|g<around*|(|\<b-x\>|)>-E<around*|{|g<around*|(|\<b-x\>|)>|}>|)><around*|(|g<around*|(|\<b-x\>|)>-E<around*|{|g<around*|(|\<b-x\>|)>|}>|)><rsup|H>|}>>,
    </with>

  where <math|g<around*|(|x|)>> function has been shown to create interesting
  higher order moments. Diagonalization of the matrix means that
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
    <math|\<b-y\>=g<rsup|-1><around*|(|\<b-W\><around*|(|\<b-x\>-E<around*|{|\<b-x\>|}>|)>+E<around*|{|g<around*|(|\<b-x\>|)>|}>|)>>
    </with>

  diagonalizes <math|\<b-G\><rsub|\<b-x\>\<b-x\>>> matrix when <math|\<b-W\>>
  diagonalizes the <math|E<around*|{|<around*|(|\<b-x\>-E<around*|{|\<b-x\>|}>|)><around*|(|\<b-x\>-E<around*|{|\<b-x\>|}>|)><rsup|H>|}>>
  matrix. That is, it is solution to linear ICA (or PCA). Additionally, it is
  interesting that inverse of <math|sinh<around*|(|x|)>> is
  <math|asinh<around*|(|x|)>> a function that is close to
  <math|tanh<around*|(|x|)>> or sigmoidal functions that have been proposed
  as non-linearities for neural networks. (Additionally,
  <math|sinh<around*|(|x|)>> has only positive coefficients in taylor
  expansion having only odd terms: <math|x<rsup|2k+1>> meaning that it is
  ``super-odd'' function as each term of polynomial form is a odd function).

  <with|font-shape|italic|NOTE: general solution to the diagonalization
  problem is <math|\<b-y\>=g<rsup|-1><around*|(|\<b-V\>*f<around*|(|\<b-x\>|)>|)>>>,
  where <math|\<b-V\>> diagonalizes <math|E<around*|{|f<around*|(|\<b-x\>|)>*f<around*|(|\<b-x\>|)><rsup|H>|}>>
  matrix and we can choose <math|f<around*|(|x|)>=g<around*|(|x|)>>.

  The whole algorithm for pre-training neural networks is then to stimulate
  neural network, collect input samples for each layer and sequentially
  calculate ICA solution for the inputs and use <math|asinh<around*|(|x|)>>
  non-linearity instead of more standard ones.\ 

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
  <math|asinh<around*|(|x|)>> has different scaling then>.

  <strong|Additions>

  Consider another case where we have output vectors <math|\<b-y\>> and lets
  define mutual joint non-linear correlation matrix\ 

  <\with|par-mode|center>
    <math|\<b-G\><rsub|\<b-x\>*\<b-y\>>=E<around*|{|<around*|(|g<around*|(|\<b-x\>|)>-E<around*|{|g<around*|(|\<b-x\>|)>|}>|)><around*|(|\<b-y\>-E<around*|{|\<b-y\>|}>|)><rsup|T>|}>>
    .
  </with>

  <\with|par-mode|center>
    <strong|<math|C<rsub|\<b-x\>*\<b-y\>>=U*S*V<rsup|T>>>
  </with>

  We can diagnonalize this matrix using non-linearities,

  <\with|par-mode|center>
    <math|\<b-x\><rprime|'>=g<rsup|-1><around*|(|\<b-U\><rsup|T>
    \<b-x\>+E<around*|{|g<around*|(|\<b-x\>|)>|}>*|)>>

    <math|\<b-y\><rprime|'>=\<b-V\><rsup|T>\<b-y\>+E<around*|{|\<b-y\>|}>*>
  </with>

  After this we can set <math|\<b-x\><rprime|'>=\<b-y\><rprime|'>> and solve
  for \ 

  <\with|par-mode|center>
    <\with|par-left|>
      <math|\<b-y\>=\<b-V\>*<around*|(|g<rsup|-1><around*|(|\<b-U\><rsup|T>
      \<b-x\>+E<around*|{|g<around*|(|\<b-x\>|)>|}>*|)>-E<around*|{|\<b-y\>|}>|)>>
    </with>
  </with>

  <emdash>

  Solution to\ 

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

    \;
  </math>

  So the overall solution to the problem

  min <math|<around*|\<\|\|\>|\<b-y\>-\<b-A\>\<b-x\>-\<b-b\>|\<\|\|\>>> is\ 

  <math|\<b-A\>=\<b-C\><rsup|T><rsub|\<b-x\>*\<b-y\>>*\<b-C\><rsup|-1><rsub|\<b-x\>*\<b-x\>>>

  <math|\<b-b\>=\<b-mu\><rsup|><rsub|\<b-y\>>-\<b-A\>\<b-mu\><rsub|\<b-x\>>>

  \;
</body>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
  </collection>
</references>