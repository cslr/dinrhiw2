<TeXmacs|1.99.12>

<\body>
  <with|font-series|bold|t-SNE algorithm implementation>

  Tomas Ukkonen 2020, <with|font-family|tt|tomas.ukkonen@novelinsight.fi>

  \;

  The algorithm is based on a t-SNE paper <with|font-shape|italic|Visualizing
  Data using t-SNE. Laurens van der Maaten and Geoffrey Hinton. Journal of
  Machine Learning Research 9 (11/2008)>. The implementation is in
  <with|font-family|tt|src/neuralnetwork/TSNE.h> and
  <with|font-family|tt|TSNE.cpp> in <with|font-family|tt|dinrhiw2>
  repository.

  \;

  We maximize KL-divergence (<math|p<rsub|i*j>> calculated from data).

  \;

  <math|D<rsub|KL><around*|(|\<b-y\><rsub|1>\<ldots\>\<b-y\><rsub|N>|)>=<big|sum><rsub|i\<neq\>j>p<rsub|i*j>*log<around*|(|<frac|p<rsub|i*j>*|q<rsub|i*j>>|)>>,
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
  Good general perplexity value is maybe 30 which we use to solve
  <math|\<sigma\><rsup|2><rsub|i>> value using bisection method.

  First we set minimum <math|\<sigma\><rsup|2><rsub|min>=0> and
  <math|\<sigma\><rsub|max><rsup|2>=trace<around*|(|\<b-Sigma\><rsub|\<b-x\>>|)>>.
  We then always select <math|\<sigma\><rsup|2><rsub|next>=<frac|\<sigma\><rsup|2><rsub|min>+\<sigma\><rsup|2><rsub|msx>|2>>
  to half the interval and calculate perplexity at
  <math|\<sigma\><rsub|next><rsup|2>> to figure out which half contains the
  target perpelexity value and stop if error is smaller than <math|0.1>.

  \;

  <with|font-series|bold|Gradient>

  \;

  We need to calculate gradient for each <math|\<b-y\><rsub|m>> in
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

  \;

  <with|font-series|bold|Optimized gradient>

  \;

  We can rewrite the gradient of <math|D<rsub|KL>> by taking partial
  derivates of distance variables <math|d<rsub|i*j>> and <math|d<rsub|j*i>>,
  <math|d<rsub|i*j>=<around*|\<\|\|\>|\<b-y\><rsub|i>-\<b-y\><rsub|j>|\<\|\|\>>>

  \;

  <math|\<nabla\><rsub|\<b-y\><rsub|i>>D<rsub|KL>=<big|sum><rsub|j><around*|(|<frac|\<partial\>D<rsub|K*L>|\<partial\>d<rsub|i*j>>*<frac|\<partial\>*d<rsub|i*j>|\<partial\>*\<b-y\><rsub|i>>+<frac|\<partial\>D<rsub|K*L>|\<partial\>d<rsub|j*i>>*<frac|\<partial\>*d<rsub|j*i>|\<partial\>*\<b-y\><rsub|i>>|)>=<big|sum><rsub|j><around*|(|<frac|\<partial\>D<rsub|K*L>|\<partial\>d<rsub|i*j>>*+<frac|\<partial\>D<rsub|K*L>|\<partial\>d<rsub|j*i>>*|)><frac|\<partial\>*d<rsub|j*i>|\<partial\>*\<b-y\><rsub|i>>=2*<big|sum><rsub|j><frac|\<partial\>D<rsub|K*L>|\<partial\>d<rsub|i*j>>*<frac|\<partial\>*d<rsub|j*i>|\<partial\>*\<b-y\><rsub|i>>*>

  \;

  The gradient of the distance variable <math|d<rsub|j*i>> is

  \;

  <math|<frac|\<partial\>*d<rsub|j*i>|\<partial\>*\<b-y\><rsub|i>>=D<around*|\<\|\|\>|\<b-y\><rsub|i>-\<b-y\><rsub|j>|\<\|\|\>>=<frac|d|d*\<b-y\><rsub|i>><sqrt|<around*|\<\|\|\>|\<b-y\><rsub|i>-\<b-y\><rsub|j>|\<\|\|\>><rsup|2>>=<frac|1|2<sqrt|><around*|\<\|\|\>|\<b-y\><rsub|i>-\<b-y\><rsub|j>|\<\|\|\>><rsup|2>>*D<around*|\<\|\|\>|\<b-y\><rsub|i>-\<b-y\><rsub|j>|\<\|\|\>><rsup|2>=<frac|\<b-y\><rsub|i>-\<b-y\><rsub|j>|*<around*|\<\|\|\>|\<b-y\><rsub|i>-\<b-y\><rsub|j>|\<\|\|\>>>*.>
  Note that the research paper gives different derivate which seems to be
  <with|font-series|bold|<with|font-shape|italic|wrong>> (?).

  Gradient of the <math|D<rsub|KL>> term is (we use auxiliary variable
  <math|Z=<big|sum><rsub|k\<neq\>l><around*|(|1+d<rsup|2><rsub|k*l>|)><rsup|-1>>).

  \;

  <\math>
    <frac|\<partial\>D<rsub|K*L>|\<partial\>d<rsub|i*j>>=-<big|sum><rsub|k\<neq\>l>*p<rsub|k*l><frac|\<partial\><around*|(|log<around*|(|q<rsub|k*l>|)>|)>|\<partial\>*d<rsub|i*j>>=-<big|sum><rsub|k\<neq\>l>*p<rsub|k*l><frac|\<partial\><around*|(|log<around*|(|q<rsub|k*l>Z|)>-log<around*|(|Z|)>|)>|\<partial\>*d<rsub|i*j>>

    =-<big|sum><rsub|k\<neq\>l>*p<rsub|k*l>*<around*|(|<frac|1|q<rsub|k*l>Z>*<frac|\<partial\><around*|(|1+d<rsup|2><rsub|k*l>|)><rsup|-1>|\<partial\>*d<rsub|i*j>>-<frac|1|Z>*<frac|\<partial\>*Z|\<partial\>*d<rsub|i*j>>|)>=2*<frac|p<rsub|i*j>|q<rsub|i*j>Z>*<around*|(|1+d<rsup|2><rsub|i*j>|)><rsup|-2>+<big|sum><rsub|k\<neq\>l>*p<rsub|k*l>*<frac|1|Z>*<frac|\<partial\>*Z|\<partial\>*d<rsub|i*j>>

    =2*p<rsub|i*j>*<around*|(|1+d<rsup|2><rsub|i*j>|)><rsup|-1>-2*<big|sum><rsub|k\<neq\>l>*p<rsub|k*l>*<frac|<around*|(|1+d<rsup|2><rsub|<rsup|>i*j>|)><rsup|-2>|Z>*

    =2*p<rsub|i*j>*<around*|(|1+d<rsup|2><rsub|i*j>|)><rsup|-1>-2*<around*|(|<frac|<around*|(|1+d<rsup|2><rsub|<rsup|>i*j>|)><rsup|-2>|Z>|)>=2*p<rsub|i*j>*<around*|(|1+d<rsup|2><rsub|i*j>|)><rsup|-1>-2**q<rsub|i*j><around*|(|1+d<rsup|2><rsub|<rsup|>i*j>|)><rsup|-1>

    =2*<around*|(|p<rsub|i*j>-q<rsub|i*j>|)><around*|(|1+d<rsup|2><rsub|<rsup|>i*j>|)><rsup|-1>
  </math>

  \;

  So we now get a simple formula for the gradient

  \;

  <math|\<nabla\><rsub|\<b-y\><rsub|i>>D<rsub|KL>=<big|sum><rsub|j>2*<around*|(|p<rsub|i*j>-q<rsub|i*j>|)><around*|(|1+d<rsup|2><rsub|<rsup|>i*j>|)><rsup|-1><around*|(|<frac|\<b-y\><rsub|i>-\<b-y\><rsub|j>|*<around*|\<\|\|\>|\<b-y\><rsub|i>-\<b-y\><rsub|j>|\<\|\|\>>>*|)>>.

  \;

  To get even better results we want to use absolute value
  <math|<around*|\||D|\|><rsub|KL>> (See later in this paper). This means we
  will compute altered gradient.

  \;

  <math|D<around*|\<\|\|\>|f<around*|(|\<b-x\>|)>|\<\|\|\>>=D<sqrt|<around*|\<\|\|\>|f<around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2>>=<frac|1|2<sqrt|<around*|\<\|\|\>|f<around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2>>>*D<around*|\<\|\|\>|f<around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2>=<frac|f<around*|(|\<b-x\>|)>|<around*|\<\|\|\>|f<around*|(|\<b-x\>|)>|\<\|\|\>>>\<nabla\>f<around*|(|\<b-x\>|)>=sign<around*|(|f<around*|(|\<b-x\>|)>|)>\<nabla\>f<around*|(|\<b-x\>|)>*>

  \;

  \;

  <\math>
    <frac|\<partial\><around*|\||D|\|><rsub|KL>|\<partial\>*d<rsub|i*j>>=<big|sum><rsub|k\<neq\>l>p<rsub|k*l>*<frac|1|\<partial\>*d<rsub|i*j>><around*|\||log<around*|(|<frac|p<rsub|k*l>|q<rsub|k*l>>|)>|\|>=-<big|sum><rsub|k\<neq\>l>sign<around*|(|log<around*|(|<frac|p<rsub|k*l>|q<rsub|k*l>>|)>|)>*p<rsub|k*l>**<frac|\<partial\><around*|(|log<around*|(|q<rsub|k*l>|)>|)>|\<partial\>*d<rsub|i*j>>
  </math>

  \;

  This means we only need to modify our gradient formula by multiplication of
  <math|sign<around*|(|x|)>> function.

  \;

  <math|\<nabla\><rsub|\<b-y\><rsub|i>><around*|\||D|\|><rsub|KL>=<big|sum><rsub|j>2**sign<around*|(|log<around*|(|<frac|p<rsub|i*j>|q<rsub|i*j>>|)>|)>*<around*|(|p<rsub|i*j>-q<rsub|i*j>|)><around*|(|1+d<rsup|2><rsub|<rsup|>i*j>|)><rsup|-1><around*|(|<frac|\<b-y\><rsub|i>-\<b-y\><rsub|j>|*<around*|\<\|\|\>|\<b-y\><rsub|i>-\<b-y\><rsub|j>|\<\|\|\>>>*|)>>.

  \;

  This optimized gradient is faster because it scales as
  <math|O<around*|(|N<rsup|2>|)>> instead of slower
  <math|O<around*|(|N<rsup|3>|)>> of the direct method.

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

  <with|font-series|bold|Improvement of the KL divergence based distribution
  comparision>

  \;

  The information theoretic distribution comparision metric <math|D<rsub|KL>>
  can be improved by using absolute values. This also symmetrices comparision
  a bit. (See my other notes about information theory/also at the end of this
  section.)

  \;

  <math|<around*|\||D|\|><rsub|KL><around*|(|\<b-y\><rsub|1>\<ldots\>\<b-y\><rsub|N>|)>=<big|sum><rsub|i\<neq\>j>p<rsub|i*j>*<around*|\||log<around*|(|<frac|p<rsub|i*j>|q<rsub|i*j>>|)>|\|>>

  \;

  Gradient of the absolute value can be computed using a simple trick.

  \;

  <math|D<around*|\<\|\|\>|f<around*|(|\<b-x\>|)>|\<\|\|\>>=D<sqrt|<around*|\<\|\|\>|f<around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2>>=<frac|1|2<sqrt|<around*|\<\|\|\>|f<around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2>>>*D<around*|\<\|\|\>|f<around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2>=<frac|f<around*|(|\<b-x\>|)>|<around*|\<\|\|\>|f<around*|(|\<b-x\>|)>|\<\|\|\>>>\<nabla\>f<around*|(|\<b-x\>|)>=sign<around*|(|f<around*|(|\<b-x\>|)>|)>\<nabla\>f<around*|(|\<b-x\>|)>*>

  \;

  This means the improved gradient is:

  \;

  <math|\<nabla\><rsub|\<b-y\><rsub|m>><around*|\||D|\|><rsub|KL>=><math|<big|sum><rsub|i\<neq\>j>p<rsub|i*j>*\<nabla\><rsub|\<b-y\><rsub|m>><around*|\||log<around*|(|<frac|p<rsub|i*j>|q<rsub|i*j>>|)>|\|>=-<big|sum><rsub|i\<neq\>j><frac|p<rsub|i*j>|q<rsub|i*j>>*sign<around*|(|log<around*|(|<frac|p<rsub|i*j>|q<rsub|i*j>>|)>|)>\<nabla\><rsub|\<b-y\><rsub|m>>q<rsub|i*j>>

  \;

  This means we only need to add <math|sign<around*|(|x|)>> non-linearity to
  the gradient calculation code. The <math|sign<around*|(|x|)>> non-linearity
  is well defined everywhere else except at zero where we can set
  <math|sign<around*|(|0|)>=1> without having much problems in practice.

  \;

  <with|font-series|bold|Justification of the modified KL divergence>

  \;

  The absolute value can be justified by following calculations. Geometric
  mean of observed symbol string is <math|P> and the number of symbols
  <math|l=1\<ldots\>L> in <math|N> symbol long string is <math|n<rsub|l>>.
  Additionally we let the length of string to go to infinity
  <math|<around*|(|N\<rightarrow\>\<infty\>|)>>:

  \;

  <math|P=<around*|(|<big|prod><rsup|N><rsub|k>p<around*|(|\<b-x\><rsub|k>|)>|)><rsup|1/N>=<around*|(|<big|prod><rsup|L><rsub|l>p<around*|(|l|)><rsub|><rsup|n<rsub|l>>|)><rsup|1/N>\<approx\><big|prod><rsup|L><rsub|l>p<around*|(|l|)><rsup|p<rsub|<around*|(|l|)>>>>

  \;

  By taking the logarithm of <math|P> we get formula for entropy:
  <math|log<around*|(|P|)>=<big|sum><rsub|l>p<around*|(|l|)>*log<around*|(|p<around*|(|l|)>|)>=-H<around*|(|L|)>>.

  \;

  Comparing distributions probabilities we can write
  (<math|N\<rightarrow\>\<infty\>>):

  \;

  <math|Q<rsub|\<b-x\>>=<around*|(|<frac|<big|prod><rsup|N><rsub|k>p<around*|(|\<b-x\><rsub|k>|)>|<big|prod><rsup|N><rsub|k>p<around*|(|\<b-y\><rsub|k>|)>>|)><rsup|1/N>=<around*|(|<big|prod><rsup|L><rsub|l><around*|(|<frac|p<rsub|\<b-x\>><around*|(|l|)>|p<rsub|\<b-y\>><around*|(|l|)>>|)><rsup|n<rsub|l>>|)><rsup|1/N>\<approx\><big|prod><rsup|L><rsub|l><around*|(|<frac|p<rsub|\<b-x\>><around*|(|l|)>|p<rsub|\<b-y\>><around*|(|l|)>>|)><rsup|p<rsub|\<b-x\>><around*|(|l|)>>>.

  \;

  And by taking the logarithm of <math|Q> we get Kullback-Leibler divergence:

  \;

  <math|log<around*|(|Q<rsub|\<b-x\>>|)>=<big|sum><rsub|l>p<rsub|\<b-x\>><around*|(|l|)>log<around*|(|<frac|p<rsub|\<b-x\>><around*|(|l|)>|p<rsub|\<b-y\>><around*|(|l|)>>|)>=D<rsub|KL>>

  \;

  Now by always taking the maximum ratio of probabilties when computing
  <math|Q> we don't have the problem that multiplication (in
  <math|<big|prod><rsup|L><rsub|l><around*|(|<frac|p<rsub|\<b-x\>><around*|(|l|)>|p<rsub|\<b-y\>><around*|(|l|)>>|)><rsup|n<rsub|l>>>-term)
  of probability ratios would cancel each other reducing the usability of
  <math|D<rsub|KL>> divergence when used for distribution comparision.

  \;

  <math|<around*|\||Q<rsub|\<b-x\>>|\|>=<around*|(|<big|prod><rsup|L><rsub|l>max<around*|(|<frac|p<rsub|\<b-x\>><around*|(|l|)>|p<rsub|\<b-y\>><around*|(|l|)>>,<frac|p<rsub|\<b-y\>><around*|(|l|)>|p<rsub|\<b-x\>><around*|(|l|)>>|)><rsup|n<rsub|l>>|)><rsup|1/N>\<approx\><big|prod><rsup|L><rsub|l>max<around*|(|<frac|p<rsub|\<b-x\>><around*|(|l|)>|p<rsub|\<b-y\>><around*|(|l|)>>,<frac|p<rsub|\<b-y\>><around*|(|l|)>|p<rsub|\<b-x\>><around*|(|l|)>>|)><rsup|p<rsub|\<b-x\>><around*|(|l|)>>>

  \;

  <math|log<around*|\||Q<rsub|\<b-x\>>|\|>=<big|sum><rsub|l>p<rsub|\<b-x\>><around*|(|l|)>log<around*|(|max<around*|(|<frac|p<rsub|\<b-x\>><around*|(|l|)>|p<rsub|\<b-y\>><around*|(|l|)>>,<frac|p<rsub|\<b-y\>><around*|(|l|)>|p<rsub|\<b-x\>><around*|(|l|)>>|)>|)>=<big|sum><rsub|l>p<rsub|\<b-x\>><around*|(|l|)><around*|\||log<around*|(|<frac|p<rsub|\<b-x\>><around*|(|l|)>|p<rsub|\<b-y\>><around*|(|l|)>>|)>|\|>=<around*|\||D<rsub|\<b-x\>>|\|><rsub|KL>>

  \;

  Further symmetrization can be done by taking the geometric mean:

  \;

  <math|<around*|\||Q|\|>=<around*|(|<around*|\||Q<rsub|\<b-x\>>|\|>*<around*|\||Q<rsub|\<b-y\>>|\|>|)><rsup|1/2>>,
  <math|log<around*|(|<around*|\||Q|\|>|)>=<frac|1|2><around*|(|log<around*|\||Q<rsub|\<b-x\>>|\|>+log<around*|\||Q<rsub|\<b-y\>>|\|>|)>=<frac|1|2><around*|(|<around*|\||D<rsub|\<b-x\>>|\|><rsub|KL>+<around*|\||D<rsub|\<b-y\>>|\|><rsub|KL>|)>>.

  \;

  \;

  <with|font-series|bold|Improvement of the MSE calculation code>

  \;

  Calculating a gradient of absolute value can be also used in minimum least
  squares (MSE) optimization where we can then easily use norm instead
  (minimum norm error - MNE) of the squared error which is then less affected
  by large outlier values.

  \;

  <math|MSE<around*|(|\<b-w\>|)>=E<rsub|\<b-x\>*\<b-y\>><around*|[|<frac|1|2><around*|\<\|\|\>|\<b-y\>-f<around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2>|]>>,
  <math|\<nabla\><rsub|\<b-w\>>*MSE<around*|(|\<b-w\>|)>=E<rsub|\<b-x\>*\<b-y\>><around*|[|<around*|(|f<around*|(|\<b-x\>|)>-\<b-y\>|)><rsup|T>\<nabla\><rsub|\<b-w\>>\<b-f\><around*|(|\<b-x\>|)>|]>>

  \;

  <math|MNE<around*|(|\<b-w\>|)>=E<rsub|\<b-x\>*\<b-y\>><around*|[|<around*|\<\|\|\>|\<b-y\>-f<around*|(|\<b-x\>|)>|\<\|\|\>>|]>>,
  <math|\<nabla\><rsub|\<b-w\>>*MNE<around*|(|\<b-w\>|)>=E<rsub|\<b-x\>*\<b-y\>><around*|[|<frac|<around*|(|f<around*|(|\<b-x\>|)>-\<b-y\>|)><rsup|T>|<around*|\<\|\|\>|f<around*|(|\<b-x\>|)>-\<b-y\>|\<\|\|\>>>\<nabla\><rsub|\<b-w\>>\<b-f\><around*|(|\<b-x\>|)>|]>>

  \;

  This means we have to just to scale the backpropagation gradient of each
  term <math|i> by dividing with <math|<around*|\<\|\|\>|\<b-y\><rsub|i>-f<around*|(|\<b-x\><rsub|i>|)>|\<\|\|\>>>.
  This means that for the large errors the effect to gradient is now smaller
  and small values have equal effect to gradient.

  \;
</body>

<initial|<\collection>
</collection>>