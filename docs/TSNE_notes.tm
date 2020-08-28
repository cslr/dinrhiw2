<TeXmacs|1.99.12>

<\body>
  <with|font-series|bold|t-SNE algorithm implementation>

  Tomas Ukkonen 2020, <with|font-family|tt|tomas.ukkonen@novelinsight.fi>

  \;

  Implementation is in <with|font-family|tt|src/neuralnetwork/TSNE.h> and
  <with|font-family|tt|TSNE.cpp> in <with|font-family|tt|dinrhiw2>
  repository.

  \;

  We maximize KL-divergence (<math|p<rsub|i*j>> calculated from data).

  \;

  <math|D<rsub|KL><around*|(|\<b-y\><rsub|1>\<ldots\>\<b-y\><rsub|N>|)>=<big|sum><rsub|i\<neq\>j>p<rsub|i*j>*log<around*|(|<frac|p<rsub|i*j>|q<rsub|i*j>>|)>>,
  <math|q<rsub|i*j>=<frac|<around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|i>-\<b-y\><rsub|j>|\<\|\|\>><rsup|2>|)><rsup|-1>|<big|sum><rsub|k><big|sum><rsub|l\<neq\>k><around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|k>-\<b-y\><rsub|l>|\<\|\|\>><rsup|2>|)><rsup|-1>>>

  \;

  The <math|p<rsub|i*j>> values are calculated from data using formulas.

  \;

  <math|p<rsub|j<around*|\||i|\<nobracket\>>>=<frac|e<rsup|-<around*|\<\|\|\>|\<b-x\><rsub|j>-\<b-x\><rsub|i>|\<\|\|\>><rsup|2>/2*\<sigma\><rsup|2><rsub|i>>|<big|sum><rsub|k\<neq\>i>e<rsup|-<around*|\<\|\|\>|\<b-x\><rsub|k>-\<b-x\><rsub|i>|\<\|\|\>><rsup|2>/2\<sigma\><rsup|2><rsub|i>>>>,
  <math|p<rsub|i<around*|\||i|\<nobracket\>>>=0>,
  <math|<big|sum><rsub|j>p<rsub|j<around*|\||i|\<nobracket\>>>=1>

  \;

  Symmetric probability values are computed from conditional probabilities
  using the formula

  <math|p<rsub|i*j>=<frac|p<rsub|j<around*|\||i|\<nobracket\>>>+p<rsub|i<around*|\||j|\<nobracket\>>>|2*N>>,<math|<big|sum><rsub|i,j>p<rsub|i*j>=1>

  \;

  The variance terms of each data point <math|\<sigma\><rsup|2><rsub|i>> is
  calculated using values <math|p<rsub|j<around*|\||i|\<nobracket\>>>> to
  search for target perplexity <math|perp<around*|(|P<rsub|i>|)>=2<rsup|H<around*|(|P<rsub|i>|)>>=2<rsup|-<big|sum><rsub|j>p<rsub|j<around*|\||i|\<nobracket\>>>*log<rsub|2><around*|(|p<rsub|j<around*|\||i|\<nobracket\>>>|)>>>.
  Good general perplexity value is maybe 30 so we need to solve
  <math|\<sigma\><rsup|2><rsub|i>> value using bisection method.

  First we set minimum <math|\<sigma\><rsup|2><rsub|min>=0> and
  <math|\<sigma\><rsub|max><rsup|2>=trace<around*|(|\<b-Sigma\><rsub|\<b-x\>>|)>>.
  We then always select <math|\<sigma\><rsup|2><rsub|next>=<frac|\<sigma\><rsup|2><rsub|min>+\<sigma\><rsup|2><rsub|msx>|2>>
  to half the interval and calculate perplexity at
  <math|\<sigma\><rsub|next><rsup|2>> to figure out which half contains the
  target perpelexity value and stop if error is smaller than <math|0.1>.

  \;

  \;

  <with|font-series|bold|Gradient>

  \;

  We need to calculate gradient for each <math|\<b-y\><rsub|i>> in
  <math|D<rsub|KL>>.

  \;

  <math|\<nabla\><rsub|\<b-y\><rsub|m>>D<rsub|KL>=\<nabla\><rsub|\<b-y\><rsub|m>>><math|<big|sum><rsub|i\<neq\>j>-p<rsub|i*j>*log<around*|(|q<rsub|i*j>|)>=-<big|sum><rsub|i\<neq\>j><frac|p<rsub|i*j>|q<rsub|i*j>>*\<nabla\><rsub|\<b-y\><rsub|m>>q<rsub|i*j>>

  \;

  The general rule to derivate <math|q<rsub|i*j>> terms is:

  \;

  <math|\<nabla\><frac|f|g>=\<nabla\>f*g<rsup|-1>=f<rprime|'>g<rsup|-2>g-f*g<rsup|-2>g<rprime|'>=<frac|f<rprime|'>g-f*g<rprime|'>|g<rsup|2>>>

  \;

  And when <math|m\<neq\>i\<neq\>j> we need to derivate only the second part

  \;

  <\math>
    *\<nabla\><rsub|\<b-y\><rsub|m\<neq\>i\<neq\>j>><around*|(|<frac|<around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|i>-\<b-y\><rsub|j>|\<\|\|\>><rsup|2>|)><rsup|-1>|<big|sum><rsub|k><big|sum><rsub|l\<neq\>k><around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|k>-\<b-y\><rsub|l>|\<\|\|\>><rsup|2>|)><rsup|-1>>|)>

    =<rsub|>-<frac|<around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|i>-\<b-y\><rsub|j>|\<\|\|\>><rsup|2>|)><rsup|-1>*|<around*|(|<big|sum><rsub|k><big|sum><rsub|l\<neq\>k><around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|k>-\<b-y\><rsub|l>|\<\|\|\>><rsup|2>|)><rsup|-1>|)><rsup|2>>\<nabla\><rsub|\<b-y\><rsub|m>><big|sum><rsub|k><big|sum><rsub|l\<neq\>k><around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|k>-\<b-y\><rsub|l>|\<\|\|\>><rsup|2>|)><rsup|-1>
  </math>

  \;

  \;

  <\math>
    \<nabla\><rsub|\<b-y\><rsub|m\<neq\>i\<neq\>j>><big|sum><rsub|k><big|sum><rsub|l\<neq\>k><around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|k>-\<b-y\><rsub|l>|\<\|\|\>><rsup|2>|)><rsup|-1>

    =\<nabla\><rsub|\<b-y\><rsub|m>><big|sum><rsub|l\<neq\>m><around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|m>-\<b-y\><rsub|l>|\<\|\|\>><rsup|2>|)><rsup|-1>+\<nabla\><rsub|\<b-y\><rsub|m>><big|sum><rsub|k\<neq\>m><around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|k>-\<b-y\><rsub|m>|\<\|\|\>><rsup|2>|)><rsup|-1>

    =2*\<nabla\><rsub|\<b-y\><rsub|m>><big|sum><rsub|l\<neq\>m><around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|m>-\<b-y\><rsub|l>|\<\|\|\>><rsup|2>|)><rsup|-1>

    =2*<big|sum><rsub|l\<neq\>m>\<nabla\><rsub|\<b-y\><rsub|m>><around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|m>-\<b-y\><rsub|l>|\<\|\|\>><rsup|2>|)><rsup|-1>

    =4*<big|sum><rsub|l\<neq\>m>-<around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|m>-\<b-y\><rsub|l>|\<\|\|\>><rsup|2>|)><rsup|-2><around*|(|\<b-y\><rsub|m>-\<b-y\><rsub|l><rsup|>*|)>
  </math>

  \;

  \;

  And when <math|y=i> or <math|y=j> we need to derivate the upper part too.

  \;

  <math|\<nabla\><rsub|\<b-y\><rsub|i>><frac|<around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|i>-\<b-y\><rsub|j>|\<\|\|\>><rsup|2>|)><rsup|-1>|<big|sum><rsub|k><big|sum><rsub|l\<neq\>k><around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|k>-\<b-y\><rsub|l>|\<\|\|\>><rsup|2>|)><rsup|-1>>=<frac|1|<big|sum><rsub|k><big|sum><rsub|l\<neq\>k><around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|k>-\<b-y\><rsub|l>|\<\|\|\>><rsup|2>|)><rsup|-1>>\<nabla\><rsub|\<b-y\><rsub|i>><around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|i>-\<b-y\><rsub|j>|\<\|\|\>><rsup|2>|)><rsup|-1>-<frac|f*g<rprime|'>|g<rsup|2>><rsub|>>

  <math|\<nabla\><rsub|\<b-y\><rsub|i>><around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|i>-\<b-y\><rsub|j>|\<\|\|\>><rsup|2>|)><rsup|-1>=-2*<around*|(|1+<around*|\<\|\|\>|\<b-y\><rsub|i>-\<b-y\><rsub|j>|\<\|\|\>><rsup|2>|)><rsup|-2>**<around*|(|\<b-y\><rsub|i>-\<b-y\><rsub|j>|)>>

  \;

  With these derivates we can then calculate derivate of <math|D<rsub|KL>>
  for each <math|\<b-y\>>. We just select step length for the gradient which
  causes increase in <math|D<rsub|KL>>.

  \;

  <with|font-series|bold|Optimization of computation>

  \;

  For large number of points the update rule is slow
  (<math|O<around*|(|N<rsup|2>|)>> scaling). Extra speed can be archieved by
  combining large away data points to a single point which is then used to
  calculate the divergence and gradient. This can be done by using
  <with|font-shape|italic|Barnes-Hut approximation> which changes
  computational complexity to near linear
  <math|O<around*|(|N*log<around*|(|N|)>|)>>.

  \;
</body>

<initial|<\collection>
</collection>>