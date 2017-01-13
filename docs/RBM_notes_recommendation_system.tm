<TeXmacs|1.0.7.15>

<style|generic>

<\body>
  <with|font-series|bold|Mathematical Notes about Recommendation System using
  RBM>

  Tomas Ukkonen, 2016, tomas.ukkonen@iki.fi

  Implementation is based on 2007 paper <em|``Restricted Boltzmann Machines
  for Collaborative Filtering''> by R. Salakhutdinov et al.

  RBM is implemented as bernoulli-bernoulli (binary) RBM because it seems to
  give better results than bernoulli-gaussian RBM although the 2007 paper
  didn't fully investigate results of gaussian-gaussian RBM.

  Visible elements of RBM for each user form a <math|K\<times\>m> sparse
  matrix <math|<with|font-series|bold|\<b-V\>>> where
  <math|\<b-V\><around*|(|k,i|)>=v<rsup|k><rsub|i>> is <math|1> if
  <math|i:th> movie is rated as <math|k> and <math|0> otherwise.

  Energy function of RBM is then

  <math|E<around*|(|\<b-V\>,\<b-h\>|)>=-<big|sum><rsub|k>\<b-v\><around*|(|k|)><rsup|>\<b-W\><around*|(|k|)>\<b-h\>-<big|sum><rsub|k>\<b-v\><rsup|><around*|(|k|)><rsup|T>\<b-a\><around*|(|k|)>-\<b-h\><rsup|T>\<b-b\>+C>

  As in normal RBM we define free energy <math|F<around*|(|\<b-V\>|)>> and
  calculate its derivates

  <\math>
    F<around*|(|\<b-V\>|)>=-log<big|sum><rsub|\<b-h\>>e<rsup|-E<around*|(|\<b-V\>,\<b-h\>|)>>
  </math>

  <math|P<around*|(|\<b-V\>|)>=<frac|1|Z>e<rsup|-F<around*|(|\<b-V\>|)>>=<frac|1|Z><big|sum><rsub|\<b-h\>>e<rsup|-E<around*|(|\<b-V\>,\<b-h\>|)>>>,
  <math|Z=<big|int>e<rsup|-F<around*|(|\<b-V\>|)>>d\<b-V\>>

  <math|<frac|\<partial\>log<around*|(|P<around*|(|\<b-V\>|)>|)>|\<partial\>\<theta\>>=-<around*|(|<frac|\<partial\>F<around*|(|\<b-V\>|)>|\<partial\>\<theta\>>-E<rsub|\<b-V\>><around*|[|<frac|\<partial\>F<around*|(|\<b-V\>|)>|\<partial\>\<theta\>>|]>|)>>

  <\math>
    F<around*|(|\<b-V\>|)>=-<big|sum><rsub|k>\<b-v\><around*|(|k|)><rsup|T>\<b-a\><around*|(|k|)>-log<big|sum><rsub|\<b-h\>>e<rsup|-<around*|(|<big|sum><rsub|k>\<b-v\><around*|(|k|)><rsup|>\<b-W\><around*|(|k|)>+\<b-b\>|)><rsup|T>\<b-h\>>

    =-<big|sum><rsub|k>\<b-v\><around*|(|k|)><rsup|T>\<b-a\><around*|(|k|)>-log<big|sum><rsub|\<b-h\>><big|prod><rsub|i>e<rsup|-<around*|(|<big|sum><rsub|k><big|sum><rsub|j>\<b-v\><rsub|j><around*|(|k|)><rsup|>\<b-W\><rsub|j*i><around*|(|k|)>+\<b-b\><rsub|i>|)><rsup|T>\<b-h\><rsub|i>>

    =-<big|sum><rsub|k>\<b-v\><around*|(|k|)><rsup|T>\<b-a\><around*|(|k|)>-<big|sum><rsub|i>log<around*|(|1+e<rsup|-<around*|(|<big|sum><rsub|k><big|sum><rsub|j>\<b-v\><rsub|j><around*|(|k|)><rsup|>\<b-W\><rsub|j*i><around*|(|k|)>+\<b-b\><rsub|i>|)>>|)>
  </math>

  Next we calculate derivates for each parameter

  <math|<frac|\<partial\>F<around*|(|\<b-V\>|)>|\<partial\>\<b-a\><around*|(|k|)>>=-\<b-v\><around*|(|k|)>>

  <math|<frac|\<partial\>F<around*|(|\<b-V\>|)>|\<partial\>\<b-b\>>=-sigmoid<around*|(|<big|sum><rsub|k>\<b-v\><around*|(|k|)><rsup|>\<b-W\><around*|(|k|)>+\<b-b\>|)>>

  <math|<frac|\<partial\>F<around*|(|\<b-V\>|)>|\<partial\>\<b-W\><around*|(|k|)>>=-\<b-v\><around*|(|k|)>*sigmoid<around*|(|<big|sum><rsub|k>\<b-v\><around*|(|k|)><rsup|>\<b-W\><around*|(|k|)>+\<b-b\>|)><rsup|T>>

  These derivates can be then plugged into generic gradient optimization
  formula for free energy functions. Optimization requires expected value of
  free energy gradient which can be approximated using contrastive
  divergence. Conditional probabilities for Gibbs sampling (CD-1) are:

  <math|P<around*|(|\<b-v\><around*|(|k|)>=1<around*|\|||\<nobracket\>>\<b-h\>|)>=<frac|exp<around*|(|\<b-b\><around*|(|k|)>+\<b-W\><around*|(|k|)>\<b-h\>|)>|<big|sum><rsub|k>exp<around*|(|\<b-b\><around*|(|k|)>+\<b-W\><around*|(|k|)>\<b-h\>|)>>>

  <math|P<around*|(|\<b-h\>=1<around*|\||\<b-V\>|\<nobracket\>>|)>=<frac|1|1+exp<around*|(|-\<b-b\>+<big|sum><rsub|k>\<b-v\><around*|(|k|)>\<b-W\><around*|(|k|)>|)>>>

  \;
</body>