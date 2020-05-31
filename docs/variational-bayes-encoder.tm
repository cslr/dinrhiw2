<TeXmacs|1.99.12>

<style|generic>

<\body>
  <with|font-series|bold|Variational Autonencoder (VAE) implementation notes>

  Tomas Ukkonen, 2020 \<less\>tomas.ukkonen@novelinsight.fi\<gtr\>

  \;

  Based on paper <with|font-shape|italic|Auto-Encoding Variational Bayes.
  Diedrik Kingma and Max Welling. 2012(?).>

  We assume we know neural network encoder
  <math|q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\<nobracket\>>|)>\<thicksim\>N<around*|(|\<b-mu\><rsub|\<b-z\>><around*|(|\<b-x\>|)>,\<b-sigma\><rsub|\<b-z\>><around*|(|\<b-x\>|)>\<b-I\>|)>>
  and decoder <math|p<rsub|\<theta\>><around*|(|\<b-x\><around*|\||\<b-z\>|\<nobracket\>>|)>\<thicksim\>N<around*|(|\<b-mu\><rsub|\<b-x\>><around*|(|\<b-z\>|)>,c*\<b-I\>|)>>
  for hidden parameters <math|\<b-z\>> as well as that we have decided
  distribution of hidden parameters <math|\<b-z\>>,
  <math|p<around*|(|\<b-z\>|)>\<thicksim\>N<around*|(|\<b-0\>,\<b-I\>|)>>

  \;

  We maximize probability of data which can be written as:

  <\math>
    log<around*|(|p<rsub|\<theta\>><around*|(|\<b-x\>|)>|)>=D<rsub|KL><around*|(|q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\<nobracket\>>|)><around*|\|||\|>p<rsub|\<theta\>><around*|(|\<b-z\><around*|\||\<b-x\>|\<nobracket\>>|)>|)>+L<around*|(|\<b-theta\>,\<b-phi\>,\<b-x\>|)>\<geqslant\>L<around*|(|\<b-theta\>,\<b-phi\>,\<b-x\>|)>

    L<around*|(|\<b-theta\>,\<b-phi\>,\<b-x\>|)>=log<around*|(|p<rsub|\<theta\>><around*|(|\<b-x\>|)>|)>+<big|int>q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\<nobracket\>>|)>*log<around*|(|<frac|p<rsub|\<theta\>><around*|(|\<b-z\><around*|\||\<b-x\>|\<nobracket\>>|)>|q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\<nobracket\>>|)>>|)>d\<b-z\>

    =<big|int><around*|(|log<around*|(|p<rsub|\<theta\>><around*|(|\<b-x\>,\<b-z\>|)>|)>-log<around*|(|*q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\<nobracket\>>|)>|)>|)>q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\<nobracket\>>|)>d\<b-z\>
  </math>

  <math|D<rsub|KL>> is positive and near constant <math|C> if distributions
  <math|p<around*|(|\<b-z\>|)>> and <math|q<around*|(|\<b-z\>|)>> are roughly
  same, so we ignore it and maximize lower bound <math|L> instead which can
  be rewritten as

  <math|L<around*|(|\<b-theta\>,\<b-phi\>,\<b-x\>|)>=-<big|int>><math|q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\<nobracket\>>|)><frac|q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\<nobracket\>>|)>|p<rsub|\<theta\>><around*|(|\<b-z\>|)>>*d\<b-z\>+<big|int>log<around*|(|p<rsub|\<theta\>><around*|(|\<b-x\><around*|\||\<b-z\>|\<nobracket\>>|)>|)>*q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\<nobracket\>>|)>*d\<b-z\>=-D<rprime|'><rsub|KL>+E<rsub|q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\|>|)>><around*|[|log<around*|(|p<rsub|\<theta\>><around*|(|\<b-x\><around*|\||\<b-z\>|\<nobracket\>>|)>|)>|]>>

  In the formula distributions are only distributions which we know so we can
  maximize it using gradient descent (<math|\<nabla\><rsub|\<b-theta\>,\<b-phi\>>L<around*|(|\<b-x\>|)>>).
  However, calculating direct gradient don't work very well because of
  high-variance of expected values/integrals so extra techniques are needed.

  \;

  Now we assume out probability distributions are gaussians so that
  analytical computations are tractable as discussed initially. The term
  <math|D<rprime|'><rsub|KL>> can be computed analytically and it is

  <\math>
    -D<rprime|'><rsub|KL>=<frac|1|2><big|sum><rsup|<rsup|D>><rsub|i=1><around*|(|1+log<around*|(|\<sigma\><rsub|\<b-z\>,i><around*|(|\<b-x\>|)><rsup|2>|)>-\<mu\><rsup|2><rsub|\<b-z\>,i>-\<sigma\><rsup|2><rsub|\<b-z\>,i>|)>=<frac|1|2>D+<frac|1|2>log<around*|(|det<around*|(|\<b-Sigma\><rsub|\<b-z\>><around*|(|\<b-x\>|)>|)>|)>-<frac|1|2><around*|\<\|\|\>|\<b-mu\><rsub|\<cal-z\>><around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2>-<frac|1|2>tr<around*|(|\<b-Sigma\><rsub|\<b-z\>><around*|(|\<b-x\>|)>|)>

    =<frac|1|2>D+<big|sum><rsup|<rsup|D>><rsub|i=1>log<around*|(|\<sigma\><rsub|\<b-z\>,i><around*|(|\<b-x\><rsup|<rsup|>>|)>|)>-<frac|1|2><around*|\<\|\|\>|\<b-mu\><rsub|\<cal-z\>><around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2>-<frac|1|2><around*|\<\|\|\>|\<sigma\><rsup|><rsub|\<b-z\>><around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2><rsup|>
  </math>

  And the gradient of the first term of lower bound <math|L> is:

  <\padded-center>
    <math|inv<around*|(|\<sigma\><rsub|\<b-z\>><around*|(|\<b-z\>|)>|)>=<around*|[|<frac|1|\<sigma\><rsub|\<b-z\>,i><around*|(|\<b-x\>|)>>|]>>
  </padded-center>

  <\padded-center>
    <math|\<nabla\><rsub|\<phi\>><around*|(|-D<rprime|'><rsub|KL>|)>=-\<b-mu\><rsup|T><rsub|\<b-z\>><around*|(|\<b-x\>|)>*\<nabla\><rsub|\<phi\>>\<b-mu\><rsub|\<b-z\>><around*|(|\<b-x\>|)>-*<around*|(|\<sigma\><rsub|\<b-z\>><around*|(|\<b-x\>|)>-inv<around*|(|\<sigma\><rsub|\<b-z\>>|)>|)><rsup|T>\<nabla\><rsub|\<phi\>>*\<sigma\><rsub|\<b-z\>><around*|(|\<b-x\>|)>>

    <math|\<nabla\><rsub|\<theta\>><around*|(|-D<rprime|'><rsub|KL>|)>=\<b-0\>>
  </padded-center>

  \;

  \;

  This leaves us with the second term

  <math|E<rsub|q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\|>|)>><around*|[|log<around*|(|p<rsub|\<theta\>><around*|(|\<b-x\><around*|\||\<b-z\>|\<nobracket\>>|)>|)>|]>=<big|int>log<around*|(|p<rsub|\<theta\>><around*|(|\<b-x\><around*|\||\<b-z\>|\<nobracket\>>|)>|)>*q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\<nobracket\>>|)>*d\<b-z\>>

  \;

  As discussed in the paper, we use reparametrization trick and rewrite
  <math|\<b-z\>=\<b-g\><rsub|\<phi\>><around*|(|\<b-varepsilon\>,\<b-x\>|)>=\<b-mu\><rsub|\<phi\>><around*|(|\<b-x\>|)>+\<b-varepsilon\>*\<sigma\><rsub|\<phi\>><around*|(|\<b-x\>|)>>
  where <math|\<b-varepsilon\>\<thicksim\>N<around*|(|\<b-0\>,\<b-I\>|)>> so
  we can rewrite integral term as <math|q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\<nobracket\>>|)>d\<b-z\>=p<around*|(|\<b-varepsilon\>|)>*d\<b-varepsilon\>>
  and sample extra variables from <math|p<around*|(|\<b-varepsilon\>|)>> and
  write:

  <\math>
    E<rsub|q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\|>|)>><around*|[|log<around*|(|p<rsub|\<theta\>><around*|(|\<b-x\><around*|\||\<b-z\>|\<nobracket\>>|)>|)>|]>=<big|int>log<around*|(|p<rsub|\<theta\>><around*|(|\<b-x\><around*|\||\<b-mu\><rsub|\<b-x\>><around*|(|\<b-z\>|)>,\<sigma\><rsub|\<b-x\>>|(>\<b-z\>|)>|)>*p<around*|(|\<b-varepsilon\>|)>*d\<b-varepsilon\>

    =<frac|1|N><big|sum><rsub|i><rsup|N>log<around*|(|p<rsub|\<theta\>><around*|(|\<b-x\><around*|\||\<b-mu\><rsub|\<b-x\>,\<theta\>><around*|(|\<b-mu\><rsub|\<b-z\>,\<phi\>><around*|(|\<b-x\>|)>+\<b-varepsilon\>*<rsub|i>\<sigma\><rsub|\<phi\>,\<b-z\>><around*|(|\<b-x\>|)>|)>,c*\<b-I\>|\<nobracket\>>|\<nobracket\>>|)>*
  </math>

  We can further calculate logarithm of the normal distribution to remove
  exponentation. Additionally, the data variance <math|\<sigma\><rsup|2>> is
  normalized to be <math|1> so <math|c=1> (<math|c=1> doesn't work well so
  variance is set to normalize error <math|c=D<rsub|\<b-x\>>>)

  <\math>
    p<around*|(|\<b-x\>|)>=<around*|(|2*\<pi\>|)><rsup|-D<rsub|x>/2>*det<around*|(|\<b-Sigma\>|)><rsup|-1/2>*exp<around*|(|-<frac|1|2><around*|(|\<b-x\>-\<b-mu\>|)><rsup|T>*\<b-Sigma\><rsup|-1><around*|(|\<b-x\>-\<b-mu\>|)>|)>
  </math>

  <math|log<around*|(|p<around*|(|\<b-x\>|)>|)>=<around*|(|-D<rsub|x>/2|)>*log<around*|(|2*\<pi\>|)>+log<around*|(|det<around*|(|\<b-Sigma\>|)><rsup|-1/2>|)>><math|-<frac|1|2><around*|(|\<b-x\>-\<b-mu\>|)><rsup|T>*\<b-Sigma\><rsup|-1><around*|(|\<b-x\>-\<b-mu\>|)>>

  <math|log<around*|(|p<around*|(|\<b-x\>|)>|)>=<around*|(|-D<rsub|x>/2|)>*log<around*|(|2*\<pi\>c|)>><math|-<frac|1|2*c><around*|(|\<b-x\>-\<b-mu\>|)><rsup|T>*<around*|(|\<b-x\>-\<b-mu\>|)>><math|>

  Now the gradients are

  <math|\<nabla\><rsub|\<theta\>>E<rsub|q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\|>|)>><around*|[|log<around*|(|p<rsub|\<theta\>><around*|(|\<b-x\><around*|\||\<b-z\>|)>|)>|]>=*<frac|1|N*c><big|sum><rsup|N><rsub|i><around*|(|\<b-x\>-\<b-mu\><rsub|\<b-x\>>|)><rsup|T>*<around*|\<nobracket\>|\<nabla\><rsub|\<theta\>>\<b-mu\><rsub|\<b-x\>,\<theta\>>|(>\<b-z\><rsub|i><around*|(|\<b-x\>|)>|)>>

  <math|\<nabla\><rsub|\<phi\>>E<rsub|q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\|>|)>><around*|[|log<around*|(|p<rsub|\<theta\>><around*|(|\<b-x\><around*|\||\<b-z\>|)>|)>|]>=*<frac|1*|N*c><big|sum><rsup|N><rsub|i><around*|(|\<b-x\>-\<b-mu\><rsub|\<b-x\>>|)><rsup|T>*<around*|\<nobracket\>|J<rsub|\<b-z\>>*\<b-mu\><rsub|\<b-x\>,\<theta\>>|(>\<b-z\><rsub|i><around*|(|\<b-x\>|)>|)>*<around*|(|\<nabla\><rsub|\<phi\>>\<b-mu\><rsub|\<b-z\>,\<phi\>><around*|(|\<b-x\>|)>+\<b-varepsilon\>*<rsub|i>\<nabla\><rsub|\<phi\>>*\<sigma\><rsub|\<phi\>,\<b-z\>><around*|(|\<b-x\>|)>|)>>

  \;

  NEEDED(?):

  To handle parameter sigma <math|\<sigma\><rsub|\<phi\>,\<b-z\>><around*|(|\<b-x\>|)>>
  better we will remap proper values from <math|<around*|[|0,\<infty\>|)>> to
  open interval <math|<around*|(|-\<infty\>,\<infty\>|)>> by using mapping
  <math|\<sigma\><around*|(|\<b-x\>|)>=e<rsup|s<around*|(|\<b-x\>|)>>>.\ 

  \;

  <\math>
    <around*|(|-D<rprime|'><rsub|KL>|)>==<frac|1|2>D+<big|sum><rsup|<rsup|D>><rsub|i=1>log<around*|(|\<sigma\><rsub|\<b-z\>,i><around*|(|\<b-x\><rsup|<rsup|>>|)>|)>-<frac|1|2><around*|\<\|\|\>|\<b-mu\><rsub|\<cal-z\>><around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2>-<frac|1|2><around*|\<\|\|\>|\<sigma\><rsup|><rsub|\<b-z\>><around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2><rsup|>

    =<frac|1|2>D+\<b-1\><rsup|T>s<around*|(|\<b-x\>|)>-<frac|1|2><around*|\<\|\|\>|\<b-mu\><rsub|\<cal-z\>><around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2>-<frac|1|2><around*|\<\|\|\>|e<rsup|\<b-s\><around*|(|\<b-x\>|)>>|\<\|\|\>><rsup|2><rsup|>

    \<nabla\><rsub|\<phi\>><around*|(|-D<rprime|'><rsub|KL>|)>=\<b-1\><rsup|T>\<nabla\><rsub|\<phi\>>s<around*|(|\<b-x\>|)>-\<b-mu\><rsub|\<cal-z\>><around*|(|\<b-x\>|)><rsup|T>\<nabla\><rsub|\<phi\>>*\<b-mu\><rsub|\<cal-z\>><around*|(|\<b-x\>|)>-e<rsup|2*\<b-s\><around*|(|\<b-x\>|)><rsup|T>>\<nabla\><rsub|\<phi\>>*e<rsup|\<b-s\><around*|(|\<b-x\>|)>><rsup|>*
  </math>

  \;

  This means our gradients will be:

  <math|\<b-z\><rsub|i><around*|(|\<b-x\>|)>=\<b-mu\><rsub|\<b-z\>,\<phi\>><around*|(|\<b-x\>|)>+diag<around*|(|\<b-varepsilon\><rsub|i>|)>*e<rsup|\<b-s\><rsub|\<phi\>><around*|(|\<b-x\>|)>>>

  <math|\<nabla\><rsub|\<theta\>>E<rsub|q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\|>|)>><around*|[|log<around*|(|p<rsub|\<theta\>><around*|(|\<b-x\><around*|\||\<b-z\>|)>|)>|]>=*<frac|1|N*c><big|sum><rsup|N><rsub|i><around*|(|\<b-x\>-\<b-mu\><rsub|\<b-x\>>|)><rsup|T>*<around*|\<nobracket\>|\<nabla\><rsub|\<theta\>>\<b-mu\><rsub|\<b-x\>,\<theta\>>|(>\<b-z\><rsub|i><around*|(|\<b-x\>|)>|)>>
  [not modified]

  <math|\<nabla\><rsub|\<phi\>>E<rsub|q<rsub|\<phi\>><around*|(|\<b-z\><around*|\||\<b-x\>|\|>|)>><around*|[|log<around*|(|p<rsub|\<theta\>><around*|(|\<b-x\><around*|\||\<b-z\>|)>|)>|]>=*<frac|1|N*c><big|sum><rsup|N><rsub|i><around*|(|\<b-x\>-\<b-mu\><rsub|\<b-x\>>|)><rsup|T>*<around*|\<nobracket\>|J<rsub|\<b-z\>>*\<b-mu\><rsub|\<b-x\>,\<theta\>>|(>\<b-z\><rsub|i><around*|(|\<b-x\>|)>|)>*<around*|(|\<nabla\><rsub|\<phi\>>\<b-mu\><rsub|\<b-z\>,\<phi\>><around*|(|\<b-x\>|)>+diag<around*|(|\<b-varepsilon\><rsub|i>*e<rsup|\<b-s\><around*|(|\<b-x\>|)>>|)>*\<nabla\><rsub|\<phi\>>*s<rsub|\<phi\>,\<b-z\>><around*|(|\<b-x\>|)>|)>>

  <math|=>

  <math|\<nabla\><rsub|\<phi\>><around*|(|-D<rprime|'><rsub|KL>|)>=\<b-1\><rsup|T>*\<nabla\><rsub|\<phi\>>\<b-s\><around*|(|\<b-x\>|)>-\<b-mu\><rsup|T><rsub|\<b-z\>><around*|(|\<b-x\>|)>*\<nabla\><rsub|\<phi\>>\<b-mu\><rsub|\<b-z\>><around*|(|\<b-x\>|)>-<around*|[|e<rsup|2*\<b-s\><around*|(|\<b-x\>|)>>|]><rsup|<rsup|T>>*<around*|[|*\<nabla\><rsub|\<phi\>>s<rsub|i><around*|(|\<b-x\>|)>|]>>

  \;

  <\math>
    <around*|(|-D<rprime|'><rsub|KL>|)>=<frac|1|2>D+<big|sum><rsup|<rsup|D>><rsub|i=1>log<around*|(|\<sigma\><rsub|\<b-z\>,i><around*|(|\<b-x\><rsup|<rsup|>>|)>|)>-<frac|1|2><around*|\<\|\|\>|\<b-mu\><rsub|\<cal-z\>><around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2>-<frac|1|2><around*|\<\|\|\>|\<sigma\><rsup|><rsub|\<b-z\>><around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2>=\<b-1\><rsup|T>*\<b-s\><around*|(|\<b-x\>|)>-<rsup|><frac|1|2><around*|\<\|\|\>|\<b-mu\><rsub|\<cal-z\>><around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2>-<frac|1|2><around*|\<\|\|\>|e<rsup|\<b-s\><around*|(|\<b-x\>|)>>|\<\|\|\>><rsup|2>
  </math>

  \;

  <with|font-series|bold|NOTE>

  As a preset encoder and decoder are preset using data
  <math|DATA=<around*|{|\<b-x\><rsub|i>|}><rsup|N><rsub|i=1>> by setting
  weight values <math|\<b-W\><around*|(|k,:|)>=\<b-x\><rsub|k><rsup|T>,k=RANDOM<around*|(|0,N-1|)><infix-and>bias
  values to zero.> The final layer is optimized using linear optimization to
  target values. The target and source values for preset encoder/decoder is
  set using PCA dimension reduction to three parameter values and encoder's
  variance/sigma values was set to constant
  <math|log<around*|(|\<sigma\>|)>=log<around*|(|0.2/sqrt<around*|(|D|)>|)>\<approx\>log<around*|(|0.115|)>,D=3>.
  This will give approximately <math|\<sigma\>=0.2> error distance in
  <math|D> dimensional space.

  <with|font-series|bold|Experiments>

  version 0.80: <math|log<around*|(|\<sigma\>|)>=log<around*|(|0.2/sqrt<around*|(|D|)>|)>\<approx\>log<around*|(|0.115|)>,D=3>
  - doesn't work so well.

  version 0.82: <math|log<around*|(|\<sigma\>|)>=log<around*|(|0.866/sqrt<around*|(|D|)>|)>\<approx\>log<around*|(|0.50|)>,D=3>:\ 

  complexity=10 worked, 10 don't work, 20 worked, 20 worked, 10 did not work,
  ..

  <with|font-series|bold|Correct error measure> / aprox loglikelihood

  Currently, VAE code (in VAE.cpp) uses reconstruction error
  <math|<big|sum><rsub|i><around*|\<\|\|\>|\<b-x\><rsub|i>-decoder<around*|(|encoder<around*|(|\<b-x\><rsub|i>|)>|)>|\<\|\|\>><rsup|2>>
  instead of log likelihood when estimating error. In practice we should use
  <math|<big|sum><rsub|i>log<around*|(|p<around*|(|\<b-x\><rsub|i>|)>|)>> so
  the error measure should be:

  <math|L<around*|(|\<b-theta\>,\<b-phi\>,\<b-x\>|)>=<frac|1|2>D<rsub|z>+\<b-1\><rsup|T>s<around*|(|\<b-x\>|)>-<frac|1|2><around*|\<\|\|\>|\<b-mu\><rsub|\<cal-z\>><around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2>-<frac|1|2><around*|\<\|\|\>|e<rsup|\<b-s\><around*|(|\<b-x\>|)>>|\<\|\|\>><rsup|2><rsup|>+<frac|1|N><big|sum><rsub|i><rsup|N>log<around*|(|p<rsub|\<theta\>><around*|(|\<b-x\><around*|\||\<b-mu\><rsub|\<b-x\>,\<theta\>><around*|(|\<b-mu\><rsub|\<b-z\>,\<phi\>><around*|(|\<b-x\>|)>+\<b-varepsilon\>*<rsub|i>\<sigma\><rsub|\<phi\>,\<b-z\>><around*|(|\<b-x\>|)>|)>,c*\<b-I\>|\<nobracket\>>|\<nobracket\>>|)>><math|>

  <math|=**C+\<b-1\><rsup|T>s<around*|(|\<b-x\>|)>-<frac|1|2><around*|\<\|\|\>|\<b-mu\><rsub|\<cal-z\>><around*|(|\<b-x\>|)>|\<\|\|\>><rsup|2>-<frac|1|2><around*|\<\|\|\>|e<rsup|\<b-s\><around*|(|\<b-x\>|)>>|\<\|\|\>><rsup|2><rsup|>-<frac|1|2N*c><big|sum><rsup|N><rsub|i>><math|<around*|\<\|\|\>|\<b-x\>-\<b-mu\><rsub|\<b-x\>,\<theta\>><around*|(|\<b-mu\><rsub|\<b-z\>,\<phi\>><around*|(|\<b-x\>|)>+\<b-varepsilon\>*<rsub|i>\<sigma\><rsub|\<phi\>,\<b-z\>><around*|(|\<b-x\>|)>|)>|\<\|\|\>><rsup|2>>

  <math|C=<frac|1|2>D<rsub|z>-<frac|1|2>D<rsub|x>*log<around*|(|2*\<pi\>*c|)>>

  Here the maximization of <math|L> means that the last term will be
  minimized (mean reconstruction error).

  \;

  TODO: add <math|\<b-c\>> variance term to the VAE.cpp code.

  TODO: we have normalized <math|\<b-x\>\<sim\>N<around*|(|0,\<sigma\><rsup|2>=1|)>>
  so <math|c=0.05> might be good target for
  <math|\<sigma\><rsub|error<around*|(|\<b-x\>|)>><rsup|2>>. This means VAE
  should have small reconstruction error which on the other side can mean bad
  generalization errror. (model complexity=20 and PCA prelearned z st.dev. is
  0.1)

  c=DIMZ/DIMX doesn't give very good results

  c=0.05 results? sound on average a bit better

  c=0.05^2*DIMZ/DIMX: results? results are mostly good (tunefish3 give's bad
  results)

  c=0.05^2 results? results are maybe better than in other cases.
  <with|font-series|bold|Use this as the default.>

  \;

  \;

  \;
</body>

<initial|<\collection>
</collection>>