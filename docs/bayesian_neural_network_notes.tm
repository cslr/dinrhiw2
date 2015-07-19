<TeXmacs|1.0.7.18>

<style|<tuple|generic|finnish>>

<\body>
  <with|font-series|bold|Uncertainty analysis for neural network>

  What I need is uncertainty analysis of neural network. Because bayesian
  method doesn't seem to work. I instead do direct uncertainty analysis.
  Let's define probability of weights through error rate

  <center|<math|p<around*|(|\<b-omega\><around*|\||data|\<nobracket\>>|)>=N<around*|(|\<b-y\><rsub|i>-\<b-f\><around*|(|\<b-x\><rsub|i>,\<b-omega\>|)><around*|\|||\<nobracket\>>0,\<b-x\><rsub|i>,\<b-y\><rsub|i>,\<sigma\><rsup|2>|)>>>

  And choose <math|\<sigma\><rsup|2>> from precomputed optimal solution
  <math|\<b-omega\><rsub|0>> which is also a starting point. The related
  energy functions are

  <\center>
    <math|U<around*|(|\<b-omega\>|)>=<frac|1|2><big|sum><rsub|i><around*|\<\|\|\>|\<b-y\><rsub|i>-f<around*|(|\<b-x\><rsub|i>,\<b-w\>|)>|\<\|\|\>><rsup|2>/\<sigma\><rsup|2>>

    <math|\<nabla\><rsub|\<b-omega\>>U<around*|(|\<b-omega\>|)>=<big|sum><rsub|i>\<nabla\><rsub|\<b-omega\>>*<frac|1|2><around*|\<\|\|\>|\<b-y\><rsub|i>-f<around*|(|\<b-x\><rsub|i>,\<b-w\>|)>|\<\|\|\>><rsup|2>/\<sigma\><rsup|2>>
  </center>

  This method works for any number of dimensions but do not have prior
  information and overfits to the data.

  <with|font-shape|italic|Implementation is in UHMC.cpp.>

  \;

  <with|font-series|bold|Hamiltonian Monte Carlo Bayesian Neural Network>
  implementation notes (<with|font-series|bold|``Bayesian Gradient
  Descent''>)

  <\with|font-shape|italic>
    This does only seem to work in 1-dimensional output for some reason.
    Implementation is in HMC.cpp.
  </with>

  Tomas Ukkonen, 2015

  Bayesian neural network extends basic feedforward network. This paper also
  describes general idea of <with|font-shape|italic|``bayesian gradient
  descent''> which arises when solving generic error minimization which
  maximizes likelihood of parameters given data instead of likelihood of data
  given parameters. This will then lead to qualitatively different and better
  solutions.

  <em|NOTE: Feedforward networks are hopelessly outdated so after you get
  this one working somehow just move as quickly as possible to Stacked RBMs
  (Bayesian GB-RBM 80% done).>

  \;

  The optimization method used is Hamiltionian Monte Carlo (HMC) which we
  assumes we have a distribution <math|<frac|1|Z>p<around*|(|\<b-w\>|)>=e<rsup|-U<around*|(|\<b-w\>|)>>>
  from which to sample from. Now according to bayesian inference rule we have

  <center|<math|p<around*|(|\<b-omega\><around*|\||data|\<nobracket\>>|)>=p<around*|(|data<around*|\||\<b-omega\>|\<nobracket\>>|)>*p<around*|(|\<b-omega\>|)>>>

  And we want to sample from <math|p<around*|(|\<b-omega\><around*|\||data|\<nobracket\>>|)>=<frac|1|Z<around*|(|data|)>>*e<rsup|-U<around*|(|\<b-omega\>|)>>>
  using HMC. We can rewrite data likelihood to be
  <math|p<around*|(|\<b-y\>,\<b-x\><around*|\||\<b-omega\>|)>|\<nobracket\>>=p<around*|(|\<b-y\><around*|\||\<b-x\>,\<b-omega\>|\<nobracket\>>|)>*p<around*|(|\<b-x\><around*|\||\<b-omega\>|\<nobracket\>>|)>>.
  We now define the first term to be squared error function with normally
  distributed noise <math|\<b-y\>\<sim\>N<around*|(|f<around*|(|\<b-x\>,\<b-w\>|)>,\<b-Iota\>*\<sigma\><rsup|2>|)>>
  with fixed variance <math|\<sigma\><rsup|2>>. For
  <math|p<around*|(|\<b-x\><around*|\||\<b-omega\>|\<nobracket\>>|)>> we must
  use crude approximation and simply regularize/simplify the problem and
  model our data to have same distribution as the data by choosing
  <math|\<b-x\>\<sim\>p<rsub|data><around*|(|\<b-x\>|)>> and select similarly
  prior for weights to be gaussian ball with unit variance
  <math|<frac|1|2><around*|\<\|\|\>|\<b-w\>|\<\|\|\>><rsup|2>>. The known
  covariance matrix <math|\<b-Sigma\>> can be obtained before sampling
  <math|\<b-omega\>> by using some pre-optimization method and calculating
  the related covariance or by using gibbs sampling to sample from
  <math|p<around*|(|\<sigma\><rsup|2><around*|\||\<b-omega\>,data|\<nobracket\>>|)>>
  before sampling from <math|p<around*|(|\<b-omega\><around*|\||\<b-Sigma\>,data|\<nobracket\>>|)>>.
  When taking logarithms from the both sides of the the inference rule this
  will lead to equation:

  <center|<math|U<around*|(|\<b-w\>|)>=<frac|1|2><big|sum><rsub|i><around*|\<\|\|\>|\<b-y\><rsub|i>-f<around*|(|\<b-x\><rsub|i>,\<b-w\>|)>|\<\|\|\>><rsup|2>/\<sigma\><rsup|2><rsup|><rsup|>+log<around*|(|Z<around*|(|\<b-omega\>|)>|)>+<frac|1|2><around*|\<\|\|\>|\<b-w\>|\<\|\|\>><rsup|2>-log<around*|(|p<rsub|data><around*|(|\<b-x\>|)>|)>+C<around*|(|data|)>>>

  The most troublesome term in the equation is the
  <math|Z<around*|(|\<b-omega\>|)>> partition function that integrates
  <with|font-shape|italic|over data>, is function of <math|\<b-omega\>> and
  cannot be typically ignored (but is ignored in some contexts/books):

  <center|<math|Z<around*|(|\<b-omega\>|)>=<big|int>e<rsup|-<frac|1|2*><big|sum><rsub|i><around*|\<\|\|\>|\<b-y\><rsub|i>-f<around*|(|\<b-x\><rsub|i>,\<b-omega\>|)>|\<\|\|\>><rsup|2>/\<sigma\><rsup|2>>p<rsub|data><around*|(|\<b-x\><rsub|1>\<ldots\>\<b-x\><rsub|N>|)>*d*\<b-x\><rsub|1>\<ldots\>d\<b-x\><rsub|N>*d\<b-y\><rsub|1>\<ldots\>d\<b-y\><rsub|N>>>

  Luckily we don't have to calculate <math|Z<around*|(|\<b-omega\>|)>>
  directly because in HMC we don't need value of
  <math|U<around*|(|\<b-omega\>|)>> but its gradient and difference (or in
  general MCMC we just need the difference):

  <\center>
    <math|\<nabla\><rsub|\<b-omega\>>U<around*|(|\<b-omega\>|)>=\<nabla\><rsub|\<b-omega\>>e<around*|(|\<b-omega\>|)>+<frac|1|Z<around*|(|\<b-omega\>|)>>\<nabla\><rsub|\<b-w\>>Z<around*|(|\<b-omega\>|)>+\<b-w\>>

    <math|U<around*|(|\<b-w\><rsub|n+1>|)>-U<around*|(|\<b-omega\><rsub|n>|)>=e<around*|(|\<b-omega\><rsub|n+1>|)>-e<around*|(|\<b-omega\><rsub|n>|)>+log<around*|(|<frac|Z<around*|(|\<b-omega\><rsub|n+1>|)>|Z<around*|(|\<b-omega\><rsub|n>|)>>|)>+<frac|1|2><around*|(|<around*|\<\|\|\>|\<b-omega\><rsub|n+1>|\<\|\|\>><rsup|2>-<around*|\<\|\|\>|\<b-omega\><rsub|n>|\<\|\|\>><rsup|2>|)>>

    <math|e<around*|(|\<b-omega\>|)>=<frac|1|2><big|sum><rsub|i>><math|<around*|\<\|\|\>|\<b-y\><rsub|i>-f<around*|(|\<b-x\><rsub|i>,\<b-w\>|)>|\<\|\|\>><rsup|2>/\<sigma\><rsup|2><rsup|><rsup|>>
  </center>

  It is rather straightforward to simplify the gradient of the partition
  function further:

  <center|<math|<frac|1|Z<around*|(|\<b-omega\>|)>>\<nabla\><rsub|\<b-w\>>Z<around*|(|\<b-omega\>|)>=-<big|int>\<nabla\><rsub|\<b-omega\>>e<around*|(|\<b-omega\>|)>*p<around*|(|data<around*|\||\<b-w\>|\<nobracket\>>|)>*d*data>>

  This last equation is interesting one because it defines sum of expected
  errors when <math|<around*|(|\<b-x\>,\<b-y\>|)>> is distributed according
  to our neural network model <math|\<b-y\>=f<around*|(|\<b-x\>,\<b-w\>|)>>.
  If we make assumption that (in our model) each pair
  <math|<around*|(|\<b-x\>,\<b-y\>|)>> is independently distributed and have
  the same distribution we can write

  <center|<math|<big|int>\<nabla\><rsub|\<b-omega\>>e<around*|(|\<b-omega\>|)>*p<around*|(|data<around*|\||\<b-w\>|\<nobracket\>>|)>*d*data=N<big|int><frac|1|2>\<nabla\><rsub|\<b-omega\>><around*|\<\|\|\>|\<b-y\>-f<around*|(|\<b-x\>,\<b-w\>|)>|\<\|\|\>><rsup|2>/\<sigma\><rsup|2>*p<rsub|model><around*|(|\<b-x\>,\<b-y\><around*|\||\<b-w\>|\<nobracket\>>|)>d*\<b-x\>*d\<b-y\>>>

  By inspecting the calculation methods we can see that gradient descent
  matches distribution of model <math|p<rsub|model><around*|(|\<b-y\><around*|\||\<b-x\>,\<b-omega\>|\<nobracket\>>|)>>
  as close as possible to distribution of data
  <math|p<rsub|data><around*|(|\<b-y\><around*|\||\<b-x\>|\<nobracket\>>|)>>.

  <with|font-series|bold|Learning variance parameter>

  In practice, variance term must be learnt in order to match model into data
  as well as possible. We can extend our model by writing\ 

  <center|<math|p<around*|(|\<b-omega\>,\<sigma\><rsup|2><around*|\||\<b-x\>,\<b-y\>|)>=p|(>\<b-omega\><around*|\||\<sigma\><rsup|2>,\<b-x\>,\<b-y\>|)>*p<around*|(|\<sigma\><rsup|2><around*|\||\<b-x\>,\<b-y\>|\<nobracket\>>|)>>>

  Weight parameters <math|\<b-omega\>> can be sampled from the first
  distribution and we can rewrite the equation another way around

  <center|<center|<math|p<around*|(|\<sigma\><rsup|2>,\<b-omega\><around*|\||\<b-x\>,\<b-y\>|)>=p|(>\<sigma\><rsup|2><around*|\||\<b-omega\>,\<b-x\>,\<b-y\>|)>*p<around*|(|\<b-omega\><around*|\||\<b-x\>,\<b-y\>|\<nobracket\>>|)>>>>

  And use again the first probability term to sample/estimate
  <math|\<sigma\><rsup|2>>. We can then alternate these sampling steps to
  generate samples from joint probability. In practice we need to estimate\ 

  <math|p<around*|(|\<sigma\><rsup|2><around*|\||\<b-omega\>,\<b-x\>,\<b-y\>|)>|\<nobracket\>>>
  which can be done simply by calculating average squared error without
  sampling (kind of maximum likelihood estimate).

  \;

  <with|font-series|bold|Analysis of <math|U<around*|(|\<b-omega\>|)>> and
  effect of <math|Z<around*|(|\<b-omega\>|)>>>

  At optimum points we have <math|\<nabla\><rsub|\<b-omega\>>U<around*|(|\<b-omega\>|)>=0>.
  Therefore we have an equation:

  <center|<math|<big|int>\<nabla\><rsub|\<b-omega\>>*e<around*|(|\<b-omega\>|)>*<around*|[|p<rsub|data><around*|(|\<b-y\><around*|\||\<b-x\>|\<nobracket\>>|)>-p<rsub|model><around*|(|\<b-y\><around*|\||\<b-x\>,\<b-omega\>|\<nobracket\>>|)>|]>*p<rsub|data><around*|(|\<b-x\>|)>*d\<b-x\>*d\<b-y\>\<thickapprox\>0>>

  So in optimum points distributions of data and model are matched, or if
  that is not possible, <math|U<around*|(|\<b-omega\>|)>> increases or
  decreases endlessly when generating sampling points using sampler. In
  practice <math|U<around*|(|\<b-omega\>|)>> tends to increase to lower
  probability meaning that <math|p<rsub|data><around*|(|\<b-y\><around*|\||\<b-x\>|\<nobracket\>>|)>\<less\>p<rsub|model><around*|(|\<b-y\><around*|\||\<b-x\>,\<b-omega\>|\<nobracket\>>|)>>.
  Additionally, we can see from the equations that
  <math|Z<around*|(|\<b-omega\>|)>> terms cause model to maximize generic
  model error while data terms minimize error of fitting function to data.
  Integrating gradient equation we can get formula for
  <math|U<rsup|><around*|(|\<b-omega\>|)>> without <math|Z> function.

  <center|<math|U<around*|(|\<b-omega\>|)>=N*<around*|(|E<rsub|data><around*|[|e<around*|(|\<b-omega\>|)>|]>-*E<rsub|model><around*|[|e<around*|(|\<b-omega\>|)>|]>|)>+<frac|1|2><around*|\<\|\|\>|\<b-omega\>|\<\|\|\>><rsup|2>>>

  This equation seem to have a problem that in practice the
  <math|E<rsub|model><around*|[|e<around*|(|\<b-omega\>|)>|]>> tends to
  dominate in our approximated models meaning that
  <math|N*E<rsub|data><around*|[|e<around*|(|\<b-omega\>|)>|]>+<frac|1|2><around*|\<\|\|\>|\<b-omega\>|\<\|\|\>><rsup|2>>
  can increase too much. In a simplified analysis there is too much variance
  in our model but in bayesian probabilistic analysis data and model has
  equal amount of variance (it might be possible to get beyond bayesian
  probability analysis somehow by altering
  <math|E<rsub|model><around*|[|e<around*|(|\<b-omega\>|)>|]>> this might
  include negative probabilities, complex number valued probabilities or
  other exotic calculation methods - altering <math|E<rsub|model>> / <math|Z>
  means that we don't anymore force our probability mass to have fixed
  value).

  We diverge when data has smaller variance (error) than modelling variance.
  Therefore we must match modelling variance to data variance but our
  modelling errors tends to increase variance/error of model. When our model
  too simple variance can grow too much.

  Assuming we have initial solution <math|\<b-omega\><rsub|0>> which is known
  to be a good function/model parameters for the data, we can write altered
  form (our model now integrates over <math|\<b-omega\>> to give a single
  good model <math|\<b-omega\><rsub|0>>):\ 

  <center|<math|U<rsup|\<ast\>><around*|(|\<b-omega\>|)>=N*<around*|(|E<rsub|data><around*|[|e<around*|(|\<b-omega\>|)>|]>-*E<rsub|model,\<b-omega\><rsub|0>><around*|[|e<around*|(|\<b-omega\>|)>|]>|)>>>

  And the negative term becomes constant because we force our model to has
  <math|\<b-omega\><rsub|0>> parameters while data term is really sum of
  squared error values (only approximating <math|E<rsub|data>>) and not given
  parameter <math|\<b-omega\><rsub|0>>. This mean we will ignore
  <math|Z<around*|(|\<b-omega\><rsub|>|)>> term which maximizes function
  complexity (variance/error). After this we will only look for solutions
  <math|\<b-omega\>> which allows small probabilistic variations in errors.
  This ignorance of <math|Z<around*|(|\<b-omega\>|)>> will also allow for
  probability mass of the function to be infinite?.\ 

  \;

  <strong|Solving inverse values of function (not needed anymore)>\ 

  If input space <math|\<b-x\>> is small we can do parallel search (fast) by
  picking <math|\<b-x\>> points randomly (or more preferrably from training
  data) and then use gradient descent to minimize error function
  <math|e<around*|(|\<b-x\>|)>=<frac|1|2><around*|(|\<b-y\>-\<b-f\><around*|(|\<b-x\>|)>|)><rsup|2>>
  and keep the solutions which are ``close enough zero'' (or at least
  <math|N> smallest error cases). This requires calculating
  <math|<frac|d\<b-f\>|d*\<b-x\>>> which we can compute using chain rule. For
  example, assuming two layer neural network layers we have:

  <\center>
    <math|\<b-f\><around*|(|\<b-x\>|)>=\<b-f\><rsub|2><around*|(|\<b-A\><rsub|2>*\<b-f\><rsub|1><around*|(|\<b-A\><rsub|1>*\<b-x\>+\<b-a\><rsub|!>|)>+\<b-a\><rsub|2>|)>>

    <math|<frac|d\<b-f\>|d\<b-x\>>=\<b-f\><rsub|2><rprime|'><around*|(|\<b-A\><rsub|2>*\<b-f\><rsub|1><around*|(|\<b-A\><rsub|1>*\<b-x\>+\<b-a\><rsub|!>|)>+\<b-a\><rsub|2>|)>*\<b-A\><rsub|2>*\<b-f\><rprime|'><rsub|1><around*|(|\<b-A\><rsub|1>\<b-x\>+\<b-a\>|)>*\<b-A\><rsub|1>>
  </center>

  \;

  And it is straightforward to extend this to multiple layers. However, the
  theoretical problem is that there are probably very large or infinite many
  possible input values <math|\<b-x\>> for many cases of <math|\<b-y\>>
  leading to very difficult to estimate <math|p<around*|(|\<b-x\><around*|\||\<b-y\>|\<nobracket\>>|)>>.
  We can limit the number of solutions if we restrict ourselves to the convex
  space (+ little extra around corners to do extrapolation..) and set up by
  training points and calculate our inverse function and probability function
  only in that space.

  In practice, the inverse solutions can be extremely large (outside the
  convex space setup by data). If we assume our data has zero mean and unit
  variance, then we can limit the solutions space by regularizing/using prior
  for <math|\<b-x\>> by forcing it to be taken from gaussian ball and define
  error function to be:

  <center|<math|E<around*|(|\<b-x\>|)>=<frac|1|2><around*|(|\<b-y\><with|font-series|bold|-\<b-f\><around*|(|\<b-x\>|)>>|)><rsup|2>+<frac|1|2><around*|\<\|\|\>|\<b-x\>|\<\|\|\>><rsup|2>/\<alpha\><rsup|2>>>

  <center|<math|\<nabla\>E<around*|(|\<b-x\>|)>=<around*|(|\<b-y\><with|font-series|bold|-\<b-f\><around*|(|\<b-x\>|)>>|)><rsup|T>\<nabla\>\<b-f\><around*|(|\<b-x\>|)>+\<b-x\>/\<alpha\><rsup|2>>>

  Where alpha is accuracy/inverse variance parameter which we can set
  <math|\<alpha\>=3>.

  After solving inverse values we can then assign probability to each
  <math|\<b-x\>> according to probability density function, normalize to
  unity and select one according to probability distribution.

  <with|font-series|bold|Solving difference equation>

  After we have computed gradient of the network we must solve ratio
  <math|log<around*|(|<frac|Z<around*|(|\<b-omega\><rsub|n+1>|)>|Z<around*|(|\<b-omega\><rsub|n>|)>>|)>>
  in order to make accept/reject decisions. We can estimate partition
  function ratios by using (reference):

  <center|<math|<frac|Z<around*|(|\<b-omega\><rsub|n+1>|)>|Z<around*|(|\<b-omega\><rsub|n>|)>>=<big|int><frac|1|Z<around*|(|\<b-omega\><rsub|n>|)>>*p*<rsup|*\<ast\>><around*|(|\<b-x\>,\<b-y\><around*|\||\<b-omega\><rsub|n+1>|\<nobracket\>>|)>*d\<b-x\>*d\<b-y\>=<big|int><frac|p<rsup|\<ast\>><around*|(|\<b-x\>,\<b-y\><around*|\||\<b-omega\><rsub|n+1>|\<nobracket\>>|)>|p*<rsup|\<ast\>><around*|(|\<b-x\>,\<b-y\><around*|\||\<b-omega\><rsub|n>|\<nobracket\>>|)>>*p<around*|(|\<b-x\>,\<b-y\><around*|\||\<b-omega\><rsub|n>|\<nobracket\>>|)>*d\<b-x\>*d\<b-y\>>>

  So we must calculate estimate <math|E<rsub|\<b-x\>,\<b-y\><around*|\||\<b-omega\><rsub|n>|\<nobracket\>>><around*|[|<frac|p<rsup|\<ast\>><around*|(|\<b-x\>,\<b-y\><around*|\||\<b-omega\><rsub|n+1>|\<nobracket\>>|)>|p*<rsup|\<ast\>><around*|(|\<b-x\>,\<b-y\><around*|\||\<b-omega\><rsub|n>|\<nobracket\>>|)>>|]>>
  by calculating ratio of biassed probability distributions of data. In
  pratice we have multiple data points in our data likelihood. This leads to
  equation:

  <center|<math|E<rsub|<around*|{|\<b-x\><rsub|i>,\<b-y\><rsub|i>|}><around*|\||\<b-omega\><rsub|n>|\<nobracket\>>><around*|[|<big|prod><rsub|i><frac|p<rsup|\<ast\>><around*|(|\<b-x\><rsub|i>,\<b-y\><rsub|i><around*|\||\<b-omega\><rsub|n+1>|\<nobracket\>>|)>|p*<rsup|\<ast\>><around*|(|\<b-x\><rsub|i>,\<b-y\><rsub|i><around*|\||\<b-omega\><rsub|n>|\<nobracket\>>|)>>|]>>>

  Now our terms are identically distributed and independent from each other
  <math|E<around*|[|A*B|]>=E<around*|[|A*|]>*E<around*|[|B|]>>, this means

  <center|<center|<math|E<rsub|<around*|{|\<b-x\><rsub|i>,\<b-y\><rsub|i>|}><around*|\||\<b-omega\><rsub|n>|\<nobracket\>>><around*|[|<big|prod><rsub|i><frac|p<rsup|\<ast\>><around*|(|\<b-x\><rsub|i>,\<b-y\><rsub|i><around*|\||\<b-omega\><rsub|n+1>|\<nobracket\>>|)>|p*<rsup|\<ast\>><around*|(|\<b-x\><rsub|i>,\<b-y\><rsub|i><around*|\||\<b-omega\><rsub|n>|\<nobracket\>>|)>>|]>=<big|prod><rsub|i>E<rsub|\<b-x\><rsub|,>\<b-y\><rsub|><around*|\||\<b-omega\><rsub|n>|\<nobracket\>>><around*|[|<frac|p<rsup|\<ast\>><around*|(|\<b-x\><rsub|>,\<b-y\><rsub|><around*|\||\<b-omega\><rsub|n+1>|\<nobracket\>>|)>|p*<rsup|\<ast\>><around*|(|\<b-x\><rsub|>,\<b-y\><rsub|><around*|\||\<b-omega\><rsub|n>|\<nobracket\>>|)>>|]>=E<rsub|\<b-x\><rsub|,>\<b-y\><rsub|><around*|\||\<b-omega\><rsub|n>|\<nobracket\>>><around*|[|<frac|p<rsup|\<ast\>><around*|(|\<b-x\><rsub|>,\<b-y\><rsub|><around*|\||\<b-omega\><rsub|n+1>|\<nobracket\>>|)>|p*<rsup|\<ast\>><around*|(|\<b-x\><rsub|>,\<b-y\><rsub|><around*|\||\<b-omega\><rsub|n>|\<nobracket\>>|)>>|]><rsup|N><with|font-series|bold|>>>>

  <center|<math|<with|font-series|bold|<frac|p<rsup|\<ast\>><around*|(|\<b-x\><rsub|>,\<b-y\><rsub|><around*|\||\<b-omega\><rsub|n+1>|\<nobracket\>>|)>|p*<rsup|\<ast\>><around*|(|\<b-x\><rsub|>,\<b-y\><rsub|><around*|\||\<b-omega\><rsub|n>|\<nobracket\>>|)>>=e<rsup|<frac|1|2><around*|\<\|\|\>|\<b-y\>-f<around*|(|\<b-x\>,\<b-w\><rsub|n>|)>|\<\|\|\>><rsup|2>/\<sigma\><rsup|2>-<frac|1|2><around*|\<\|\|\>|\<b-y\>-f<around*|(|\<b-x\>,\<b-w\><rsub|n+1>|)>|\<\|\|\>><rsup|2>/\<sigma\><rsup|2>>>>>

  \;

  And we sample from <math|p<rsub|model><around*|(|\<b-y\>,\<b-x\><around*|\||\<b-w\><rsub|n>|\<nobracket\>>|)>=p<rsub|model><around*|(|\<b-y\><around*|\||\<b-x\>,\<b-w\><rsub|n>|\<nobracket\>>|)>*p<rsub|data><around*|(|\<b-x\>|)>>
  in feedforward neural network because we are only interested in conditional
  distribution <math|p<around*|(|\<b-y\><around*|\||\<b-x\>|\<nobracket\>>|)>>
  which is in our model is normal distribution.

  \;

  <with|font-series|bold|Additional implementation notes>

  In practice, we need to estimate negative phase gradient and zratio by
  sampling from distribution <math|p<rsub|model><around*|(|\<b-x\>,\<b-y\><around*|\||\<b-w\>|\<nobracket\>>|)>>,
  which we can only crudely approximate. We are estimating mean value of
  z-ratio and mean value of gradients (vector) and we much estimate error
  levels in both of them in order to gather enough samples.

  TODO

  <with|font-series|bold|Divergence of HMC sampler>

  In practice HMC sampler diverge to larger and larger error levels after
  successfully visiting small levels of error. The reason for this is
  currently unclear but only way for sampler to keep accepting the new
  samples is that Z-ratio becomes very small to always dominate the results
  which then also means that gradient and is based on Z-value and other terms
  are ignored.

  More detailed analysis seems to imply that the problem is variance (or
  covariance) term in bayesian neural network. If variance is set to unity,
  it sets the accuracy (level of detail) of our predicted outcome. Too high
  variance when compared to data variance means that model ``underfits'' to
  the data. Moreover, gradient descent rule now keeps increasing
  <math|U<around*|(|\<b-w\>|)>> leading to flat posterior distribution
  <math|p<around*|(|\<b-w\>|)>> [because any parameters can fit to data with
  too much variance in model] and high model error levels. This is because
  tails of the model distribution (extreme values) are more likely in
  <math|p<rsub|model><around*|(|\<b-x\>,\<b-y\><around*|\||\<b-w\>|\<nobracket\>>|)>>
  which means that following the negative gradient causes
  <math|U<around*|(|\<b-w\>|)>> to increase

  <center|<with|font-series|bold|<math|\<nabla\><rsub|\<b-omega\>>U<around*|(|\<b-omega\>|)>=<big|int>\<nabla\><rsub|\<b-omega\>><frac|1|2><around*|\<\|\|\>|\<b-y\>-\<b-f\><around*|(|\<b-y\><around*|\||\<b-omega\>|\<nobracket\>>|)>|\<\|\|\>><rsup|2><around*|[|p<rsub|data><around*|(|\<b-y\><around*|\||\<b-x\>|\<nobracket\>>|)>*p<rsub|data><around*|(|\<b-x\>|)>-p<rsub|model><around*|(|\<b-y\><around*|\||\<b-x\>,\<b-w\>|\<nobracket\>>|)>*p<rsub|data><around*|(|\<b-x\>|)><rsub|>|]>+\<b-w\>>>>

  Additionally, we can also see that gradient descent rule will converge when
  model distribution has converged to data distribution. And because in
  practice we are only interested in <math|p<around*|(|\<b-y\><around*|\||\<b-x\>|\<nobracket\>>|)>>
  distribution we don't have to calculate inverse of
  <math|\<b-f\><around*|(|\<b-x\>|)>> in order to try to (hopelessly) match
  the whole distribution but sample from <math|p<rsub|data><around*|(|\<b-x\>|)>>
  and then from <math|p<rsub|model><around*|(|\<b-y\><around*|\||\<b-x\>,\<b-omega\>|\<nobracket\>>|)>>
  in order to generate negative particles. So in order for unit variance
  method to work, our data must have <with|font-series|bold|larger> variance
  than unit variance and ``the level of detail'' in our data must have unit
  st.dev.

  In order to fix this, we must update variance parameter
  <math|\<sigma\><rsup|2>> at every sampling step using Gibbs sampling. First
  we have <math|\<b-w\>> and we sample from
  <math|p<around*|(|\<sigma\><rsup|2><around*|\||\<b-omega\>|\<nobracket\>>,data|)>\<propto\>p<around*|(|data<around*|\||\<b-w\>,\<sigma\><rsup|2>|\<nobracket\>>|)>*p<around*|(|\<sigma\><rsup|2>|)>>.
  After we have sampled from <math|\<sigma\><rsup|2>> we use our previous
  results to sample using fixed <math|\<sigma\><rsup|2>> from
  <math|\<b-p\><around*|(|\<b-w\><around*|\||data,\<sigma\><rsup|2>|\<nobracket\>>|)>>
  and store our sample <math|<around*|(|\<b-omega\><rsub|i>,\<b-sigma\><rsub|i><rsup|2>|)>>.

  We can directly sample from distribution

  <center|<math|p<around*|(|\<sigma\><rsup|2><around*|\||\<b-omega\>|\<nobracket\>>,data|)>\<thicksim\>*X<rsup|2><around*|(|n*d-1,s<rsup|2>|)>>,
  s<math|<rsup|2>=<big|sum><rsub|i><around*|\<\|\|\>|\<b-y\><rsub|i>-\<b-f\><around*|(|\<b-x\><rsub|i>,\<b-w\>|)>|\<\|\|\>><rsup|2>/<around*|(|n*d|)>>><math|<with|font-series|bold|\<nocomma\>>>

  We can sample from this by generating <math|n*d-1>, normally distributed
  zero mean unit variance variables and calculating their squared sum.
  <math|*Z=<big|sum><rsub|i>X<rsup|2><rsub|i>>, after this we scale by
  <math|s<rsup|2>>.

  <with|font-series|bold|Sampling covariance matrix <math|\<b-Sigma\>>>

  Similarly, we can do calculate posterior distribution
  <math|p<around*|(|\<b-Sigma\><around*|\||data,\<b-omega\>|\<nobracket\>>|)>>
  for covariance matrix (InvWishart distribution) and sample from it by using
  standard multivariate normal distribution inference.

  The prior for the normally distributed zero mean data
  <math|N<around*|(|\<b-y\><rsub|i>-\<b-f\><around*|(|\<b-x\><rsub|i>,\<b-w\>|)><around*|\||\<b-mu\>,\<b-Sigma\>|\<nobracket\>>|)>>
  is zero mean normal distribution <math|\<b-mu\>\<sim\>N<around*|(|\<b-mu\><rsub|0>,\<b-Sigma\>/\<kappa\><rsub|0>|)>>
  and <math|\<b-Sigma\><rsup|-1>\<sim\>Wishart<rsub|\<b-v\><rsub|0>><around*|(|\<b-Lambda\><rsub|0><rsup|-1><rsup|>|)>>.
  For mean we decide <math|\<b-mu\><rsub|0>=0> and
  <math|\<kappa\><rsub|0>=\<infty\>> and we get a posterior distribution for
  <math|\<b-Sigma\><around*|\||data,\<b-omega\>|\<nobracket\>>\<sim\>InvWishart<rsub|v<rsub|n>><around*|(|\<b-Lambda\><rsup|-1><rsub|n>|)>>.

  <math|v<rsub|n>=v<rsub|0>+n>

  <math|\<b-Lambda\><rsub|n>=\<b-Lambda\><rsup|-1><rsub|0>+<big|sum><rsub|i>\<b-z\><rsub|i>\<b-z\><rsup|T><rsub|i>>
  , <math|\<b-z\><rsub|i>=\<b-y\><rsub|i>-\<b-f\><around*|(|\<b-x\><rsub|i>,\<b-omega\>|)>>

  And we can get non-informative prior by taking parameter values
  <math|\<b-Lambda\><rsub|0>=0> and <math|v<rsub|0>=-1>. Now we need to
  sample from InvWishart distribution. This means
  <math|\<b-Sigma\><rsup|-1><rsup|><rsup|>\<sim\>Wishart<rsub|\<b-v\><rsub|n>><around*|(|\<b-Lambda\><rsub|n><rsup|-1>*|)>>
  and we sample from Wishart distribution. This can be done by sampling
  <math|\<b-alpha\><rsub|i>\<sim\>N<around*|(|\<b-0\>,\<b-Lambda\><rsup|-1><rsub|n>|)>>
  and calculating <math|<big|sum><rsup|\<b-v\><rsub|n>><rsub|i>\<b-alpha\><rsub|i>*\<b-alpha\><rsup|T><rsub|i>>.
  (<with|font-shape|italic|Bayesian Data Analysis. Andrew Gelman. Appendix
  A.>)

  When number of observations is less than <math|dim<around*|(|\<b-z\>|)>>:
  <math|n\<leqslant\>dim<around*|(|\<b-z\>|)>> we set identity matrix prior
  <math|v<rsub|0>=dim<around*|(|\<b-z\>|)>-n><math|\<nocomma\>>,
  <math|\<b-Lambda\><rsub|0>=\<b-I\>> and otherwise we use non-informative
  prior <math|\<b-Lambda\><rsub|0>=0> and <math|v<rsub|0>=-1>.

  In practice we choose <math|\<sigma\><rsup|2>=min<around*|(|eig<around*|(|\<b-Sigma\>|)>|)>>.

  \;

  <strong|Calculating backpropagration gradient of error term>

  Calculating gradient of error term <math|\<nabla\><rsub|\<b-omega\>>*e<around*|(|\<b-omega\>|)>>
  is complicated because the basic backprogragation algorithm calculates
  gradient minimizing non whitened error term. The error function:\ 

  <center|<math|e<around*|(|\<b-omega\>|)>=<frac|1|2><big|sum><rsub|i>><math|<around*|(|\<b-y\><rsub|i>-\<b-f\><around*|(|\<b-x\><rsub|i>,\<b-w\>|)>|)><rsup|T>\<b-Sigma\><rsup|-1><around*|(|\<b-y\><rsub|i>-\<b-f\><around*|(|\<b-x\><rsub|i>,\<b-w\>|)>|)><rsup|><rsup|>>>

  can be rewritten using <math|\<b-W\>=\<b-D\><rsup|1/2>*\<b-X\><rsup|T>>
  where <math|\<b-Sigma\><rsup|-1>=\<b-X\>*\<b-D\>*\<b-X\><rsup|T>>,\ 

  <\center>
    <math|e<around*|(|\<b-omega\>|)>=<frac|1|2><big|sum><rsub|i>><math|<around*|(|\<b-W\>*\<b-y\><rsub|i>-\<b-W\>*\<b-f\><around*|(|\<b-x\><rsub|i>,\<b-w\>|)>|)><rsup|T><around*|(|\<b-W\>*\<b-y\><rsub|i>-\<b-W\>*\<b-f\><around*|(|\<b-x\><rsub|i>,\<b-w\>|)>|)><rsup|><rsup|>>

    <center|<math|<wide|e|~><around*|(|\<b-omega\>|)>=<frac|1|2><big|sum><rsub|i>><math|<around*|(|<wide|\<b-y\>|~><rsub|i>-<wide|\<b-f\>|~><around*|(|\<b-x\><rsub|i>,\<b-w\>|)>|)><rsup|T><around*|(|<wide|*\<b-y\>|~><rsub|i>-<wide|\<b-f\>|~><around*|(|\<b-x\><rsub|i>,\<b-w\>|)>|)><rsup|><rsup|>>>
  </center>

  Which have preferred squared error form. However, our neural network now
  has one extra layer <strong|W> which scales and rotates actual output of
  the function. Therefore we must look into backpropagation algorithm and
  make slight modifications to calculate gradient when the output layer is
  changed with additional extra linear layer.

  Update rule for local gradient and the update rule for the weights is:\ 

  <math|\<b-delta\><rsub|l-1><rsup|>=diag<around*|(|<with|font-series|bold|\<b-varphi\><rprime|'><around*|(|\<b-v\><rsub|l-1>|)>>|)>*\<b-W\><rsup|T><rsub|l>*\<b-delta\><rsub|l>*>,
  <math|\<b-W\><rsup|l-1><rsub|j*i>=\<b-delta\><rsub|>*\<b-y\><rsup|T><rsub|>>

  \;

  \;

  <math|>

  \;

  \;

  \;
</body>

<initial|<\collection>
</collection>>