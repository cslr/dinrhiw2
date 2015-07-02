<TeXmacs|1.99.2>

<style|<tuple|generic|finnish>>

<\body>
  <with|font-series|bold|Hamiltonian Monte Carlo Bayesian Neural Network>
  implementation notes

  Tomas Ukkonen, 2015

  Bayesian neural network extends basic feedforward network to better handle
  uncertainty in the data. It does not typically significally improve results
  of the basic feedforward network when trained using L-BFGS or other
  reasonable optimization methods. NOTE: the framework used fits quite well
  to <with|font-shape|italic|any function> (but the relationships in the data
  may not be functions! <math|x> can have multiple good outputs <math|y>)
  that we try to extend to better handle multiple different outputs or
  uncertainty.

  The optimization method used is Hamiltionian Monte Carlo (HMC) which we
  assumes we have a distribution <math|<frac|1|Z>p<around*|(|\<b-w\>|)>=e<rsup|-U<around*|(|\<b-w\>|)>>>
  from which to sample from. Now according to bayesian inference rule we have

  <center|<math|p<around*|(|\<b-omega\><around*|\||data|\<nobracket\>>|)>=p<around*|(|data<around*|\||\<b-omega\>|\<nobracket\>>|)>*p<around*|(|\<b-omega\>|)>/p<around*|(|data|)>>>

  And we want to sample from <math|p<around*|(|\<b-omega\><around*|\||data|\<nobracket\>>|)>=<frac|1|Z>e<rsup|-U<around*|(|\<b-omega\>|)>>>
  using HMC. We now define data likelihood to be squared error function with
  normally distributed noise <math|\<b-y\>\<sim\>N<around*|(|f<around*|(|\<b-x\>,\<b-w\>|)>,\<sigma\><rsup|2>|)>>
  with fixed variance (<math|\<sigma\>=1>) and similarly select prior for
  weights to be gaussian ball with unit variance
  <math|<frac|1|2><around*|\<\|\|\>|\<b-w\>|\<\|\|\>><rsup|2>>. When taking
  logarithms from the both sides of the the inference rule this will lead to
  equation:

  <center|<math|U<around*|(|\<b-w\>|)>=<frac|1|2><big|sum><rsub|i><around*|\<\|\|\>|\<b-y\><rsub|i>-f<around*|(|\<b-x\><rsub|i>,\<b-w\>|)>|\<\|\|\>><rsup|2>+log<around*|(|Z<around*|(|\<b-omega\>|)>|)>+<frac|1|2><around*|\<\|\|\>|\<b-w\>|\<\|\|\>><rsup|2>+C<around*|(|data|)>>>

  The most trouble some term in the equation is the
  <math|Z<around*|(|\<b-omega\>|)>> partition function that integrates
  <with|font-shape|italic|over data>, is function of <math|\<b-omega\>> and
  cannot be typically ignored (but is in some contexts/books):

  <center|<math|Z<around*|(|\<b-omega\>|)>=<big|int>e<rsup|-<frac|1|2*><around*|\<\|\|\>|\<b-y\><rsub|i>-f<around*|(|\<b-x\><rsub|i>,\<b-omega\>|)>|\<\|\|\>>>*d*\<b-x\><rsub|1>\<ldots\>d\<b-x\><rsub|N>*d\<b-y\><rsub|1>\<ldots\>d\<b-y\><rsub|N>>>

  Luckily we don't have to calculate <math|Z<around*|(|\<b-omega\>|)>>
  directly because in HMC we don't need value of
  <math|U<around*|(|\<b-omega\>|)>> but its gradient and difference (or in
  general MCMC we just need the difference):

  <\center>
    <math|\<nabla\><rsub|\<b-omega\>>U<around*|(|\<b-omega\>|)>=\<nabla\><rsub|\<b-omega\>>e<around*|(|\<b-omega\>|)>+<frac|1|Z<around*|(|\<b-omega\>|)>>\<nabla\><rsub|\<b-w\>>Z<around*|(|\<b-omega\>|)>+\<b-w\>>

    <math|U<around*|(|\<b-w\><rsub|n+1>|)>-U<around*|(|\<b-omega\><rsub|n>|)>=e<around*|(|\<b-omega\><rsub|n+1>|)>-e<around*|(|\<b-omega\><rsub|n>|)>+log<around*|(|<frac|Z<around*|(|\<b-omega\><rsub|n+1>|)>|Z<around*|(|\<b-omega\><rsub|n>|)>>|)>+<around*|(|\<b-omega\><rsub|n+1>-\<b-omega\><rsub|n>|)>>

    <math|e<around*|(|\<b-omega\>|)>=<frac|1|2><big|sum><rsub|i><around*|\<\|\|\>|\<b-y\><rsub|i>-f<around*|(|\<b-x\><rsub|i>,\<b-w\>|)>|\<\|\|\>><rsup|2>>
  </center>

  It is rather straightforward to simplify the gradient of the partition
  function further:

  <center|<math|<frac|1|Z<around*|(|\<b-omega\>|)>>\<nabla\><rsub|\<b-w\>>Z<around*|(|\<b-omega\>|)>=-<big|int>\<nabla\><rsub|\<b-omega\>>e<around*|(|\<b-omega\>|)>*p<around*|(|data<around*|\||\<b-w\>|\<nobracket\>>|)>*d*data>>

  This last term is interesting one because it defines sum of expected errors
  when <math|<around*|(|\<b-x\>,\<b-y\>|)>> is distributed according to our
  neural network model <math|\<b-y\>=f<around*|(|\<b-x\>,\<b-w\>|)>>. If we
  make assumption that (in our model) each pair
  <math|<around*|(|\<b-x\>,\<b-y\>|)>> is independently distributed and have
  the same distribution we can write

  <center|<math|<big|int>\<nabla\><rsub|\<b-omega\>>e<around*|(|\<b-omega\>|)>*p<around*|(|data<around*|\||\<b-w\>|\<nobracket\>>|)>*d*data=N<big|int><frac|1|2>\<nabla\><rsub|\<b-omega\>><around*|\<\|\|\>|\<b-y\>-f<around*|(|\<b-x\>,\<b-w\>|)>|\<\|\|\>><rsup|2>*p<rsub|model><around*|(|\<b-x\>,\<b-y\><around*|\||\<b-w\>|\<nobracket\>>|)>d*\<b-x\>*d\<b-y\>>>

  These calculations has been similar to RBM machine probability calculations
  so the same idea can be used to approximate the integral over model
  distribution by using contrastive divergence: 1) we start with randomly
  selected <math|\<b-x\>> from the training set and then sample
  <math|\<b-y\>> from the normal distribution
  <math|N<around*|(|f<around*|(|\<b-x\>,\<b-w\>|)>,\<sigma\><rsup|2>|)>>, 2)
  after this 2) we should sample <math|\<b-cal-x\>> from the distribution
  <math|p<rsub|model><around*|(|\<b-x\><around*|\||\<b-y\>|\<nobracket\>>|)>>
  but in practice this is impossible in our model - unless we can calculate
  inverse of <math|f<around*|(|\<b-x\>,\<b-w\>|)>> in which case we would
  have impulse distribution at <math|f<rsup|-1><around*|(|\<b-y\>|)>> which
  should used.

  <with|font-series|bold|Deep network principle>

  This result then motivates us to train two neural networks at the same
  time, direct mapping from <math|\<b-x\>> to <math|\<b-y\>> and inverse
  mapping from <math|\<b-y\>> to <math|\<b-x\>> and learn the parameters
  <math|\<b-w\><rsub|\<b-x\>*\<b-y\>>> and
  <math|\<b-w\><rsub|\<b-y\>*\<b-x\>>> at the same time (``same model'') so
  that we can then use each other to calculate conditional distributions
  <math|p<around*|(|\<b-y\><around*|\||\<b-x\>|)>|\<nobracket\>>> and
  <math|p<around*|(|\<b-x\><around*|\||\<b-y\>|\<nobracket\>>|)>> at the same
  time. We have therefore two interlocked gradient descent formulas:

  <\center>
    <math|\<nabla\><rsub|\<b-omega\><rsub|1>>U<around*|(|\<b-omega\><rsub|1>|)>=\<nabla\><rsub|\<b-omega\><rsub|1>>e<rsub|1><around*|(|\<b-omega\><rsub|1>|)>+<frac|1|Z<around*|(|\<b-omega\><rsub|1>|)>>\<nabla\><rsub|\<b-w\><rsub|1>>Z<around*|(|\<b-omega\><rsub|1>|)>+\<b-w\><rsub|1>>

    <math|\<nabla\><rsub|\<b-omega\><rsub|2>>U<around*|(|\<b-omega\><rsub|2>|)>=\<nabla\><rsub|\<b-omega\><rsub|2>>e<around*|(|\<b-omega\><rsub|2>|)>+<frac|1|Z<around*|(|\<b-omega\><rsub|2>|)>>\<nabla\><rsub|\<b-w\><rsub|2>>Z<around*|(|\<b-omega\><rsub|2>|)>+\<b-w\><rsub|2>>
  </center>

  These are independent from each other, <with|font-series|bold|except>, that
  when calculating ``negative'' phase of the gradient descent both of the
  current values of <math|\<b-omega\><rsub|1>> and <math|\<b-omega\><rsub|2>>
  are used to generate negative particles which are then used by compute real
  gradient of both gradient descent algorithms. <with|font-shape|italic|This
  is interesting result because I haven't seen this kind of principle or idea
  in optimization literature anywhere else than in the context of RBMs>.
  Moreo<strong|<em|w>>ver, we know that in practice <math|\<b-w\><rsub|1>>
  and <math|\<b-omega\><rsub|2>> are not indepedent but related to each other
  somehow so, ideally, this more detailed interaction between variables
  should be taken into account when calculating datalikelyhood probability.

  \;

  \;

  \;
</body>