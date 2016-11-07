<TeXmacs|1.0.7.15>

<style|generic>

<\body>
  <strong|Some notes about BB- and GB-RBM implementation>

  Tomas Ukkonen, 2015 <verbatim|tomas.ukkonen@iki.fi>

  <strong|Restricted Boltzmann machines> are tricky to implement as there
  doesn't seem to be very good reference material around describing all the
  details of the implementation. Here are some notes about RBM implementation
  in <verbatim|..src/neuralnetwork/RBM.h> and
  <verbatim|..src/neuralnetwork/RBM.cpp>. The learning method to be used is
  CD-10.

  I couldn't find update rules for bias-terms used to calculate approximate
  gradient descent for (contrastive divergence) so I used the following
  approach:

  <\itemize-dot>
    <item>there are no separate bias-terms

    <item>instead visible <math|\<b-v\>> and hidden <strong|<math|\<b-h\>>>
    states are always <em|extended> to have one extra <math|1> after the data
    terms, so the actual terms given to the RBM are
    <math|<around*|[|\<b-v\>\<nocomma\>,1|]>> and
    <math|<around*|[|\<b-h\>,1|]>>
  </itemize-dot>

  This then makes the calculations relatively simple as you will only have to
  care about the weight matrix <math|\<b-W\>> and can ignore other terms. How
  to handle these equal to <math|1> states during the stimulation phase of
  RBM in CD? The correct approach seems to handle them similarly to all other
  variables, in other words we just simulate and simulate their states
  normally. This then leads to situation where the bias terms automatically
  find rules that always set those values to <math|1>.

  \;

  <strong|TOY PROBLEM ANALYSIS>

  The function <verbatim|rbm_test()> in <verbatim|src/neuralnetwork/tst/test.cpp>
  generates a toy problem for which RBM CD-10 algorithm is tested. The data
  is generated as follows:

  <\itemize>
    <item>hidden states (2) are selected randomly between 0 and 1

    <item>after this visible states (4) are generated from hidden states by
    directly setting the <math|k>:th state exactly same as the <math|k/2>:th
    state in a hidden state

    <item>in practice <math|<around*|[|0,1|]>\<Rightarrow\><around*|[|0,0,1,1|]>>,<math|<around*|[|1,0|]>\<Rightarrow\><around*|[|1,1,0,0|]>>,
    <math|<around*|[|1,1|]>\<Rightarrow\><around*|[|1,1,1,1|]>> and
    <math|<around*|[|0,0|]>\<Rightarrow\><around*|[|0,0,0,0|]>>

    <item>so the relationship between hidden states and input states is
    really simple, the easiest way is to simply to calculate mean value of
    every <math|2> items in visible layer and used it as a hidden layer
  </itemize>

  The results of for running CD-10 algorithm which discretizes or samples all
  values to <math|0> and <math|1> are always the same, the weight matrix for
  hidden to visible layers have the form:

  <center|<math|\<b-W\>=><block|<tformat|<table|<row|<cell|4.6>|<cell|4.4>|<cell|4.3>|<cell|4.5>|<cell|5.6>>|<row|<cell|0.0>|<cell|-0.1>|<cell|-9.7>|<cell|-9.8>|<cell|8.9>>|<row|<cell|-9.7>|<cell|-9.7>|<cell|0.09>|<cell|0.2>|<cell|8.7>>>>>>

  This weight matrix clearly generates three hidden states where the first
  one is equal to the state <math|1> which is always on. Because binary
  values only take values <math|0> and <math|1> the sigmoid function only
  gets positive values (the last term is <math|5.6> meaning that the all
  <math|1> term in input layer sets value to be extremely likely to be
  <math|1> even when other terms are <math|0>). After this, the second term
  sets the term to be <math|0> when the second part of the inputs are
  <math|1> and <math|0> when the inputs are <math|1> because in this case the
  sigmoidal value will be negative. And similarly for the last term.

  Given this intepretation, the reversed weight matrix <math|\<b-W\><rsup|T>>
  makes now also sense:

  <center|<math|\<b-W\><rsup|T>=><center|<block|<tformat|<table|<row|<cell|4.6>|<cell|+0.1>|<cell|-9.7>>|<row|<cell|4.4>|<cell|-0.1>|<cell|-9.7>>|<row|<cell|4.3>|<cell|-9.7>|<cell|0.0>>|<row|<cell|4.5>|<cell|-9.7>|<cell|0.2>>|<row|<cell|5.6>|<cell|8.9>|<cell|8.7>>>>>>>

  If we keep in mind that the first term is always <math|1> in our reversed.
  Then the first two terms are <math|1> only when the third term is zero and
  zero only when the third term is one (negative sigmoid value). Similar is
  true for the second set of variables and finally the 5th variable is always
  one due to fact that state values are always non-negative meaning that
  sigmoidal value will be always be positive.

  <em|This then proofs and shows that the toy problem is properly solved by
  <strong|<em|the binary RBM with the CD-10>> learning rule. It can find
  proper two states and the one artificial always one state using the simple
  contrastive divergence algorithm.> This shows RBM is capable of data
  reduction.

  \;

  <with|font-series|bold|Energy based model of RBM>

  The continuous RBM is more complicated to implement and understand. It
  seems that Gaussian-Bernoulli RBM could be the right choice (or maybe
  Beta-Bernoulli RBM) to model continuous input values. Because of this, I
  try to derive the whole BB-RBM and then continuous GB-RBM theory from
  energy based models (EBMs). It is important(?) to remember that
  Bernoulli-Bernoulli RBM model is related to Ising models and statistical
  physics but hidden variable models in general are not.

  The energy of the restricted boltzman machine is:\ 

  <\math>
    E<rsub|B*B><around|(|v,h|)>=a<rsup|T>v+v<rsup|T>W*h+b<rsup|T>h

    E<rsub|G*B><around|(|v,h|)>=<frac|1|2><around|\<\|\|\>|v-a|\<\|\|\>><rsup|2>+v<rsup|T>W*h+b<rsup|T>h
  </math>

  Note that we could have added additional constant <math|c> term to these
  equations but it is not needed because probability distributions are
  normalized to have unit probability mass.

  Now the probability of the <math|<around|(|v,h|)>> and only observed
  variables <math|v> is:

  <math|P<around|(|v,h|)>=<frac|1|Z>*e<rsup|-E<around|(|v,h|)>>>,
  <math|Z=<big|sum><rsub|v,h>e<rsup|-E<around|(|v,h|)>>>

  <math|P<around|(|v|)>=<frac|1|Z><big|sum><rsub|h>e<rsup|-E<around|(|v,h|)>>>

  Now the hidden variables are Bernoulli distributed, that is, they take only
  values <math|0> and <math|1> which is a strong regularizer for the system.

  We want to calculate probabilities of hidden <math|h> and visible <math|v>
  neurons:\ 

  <math|P<rsub|B*B><around|(|h\|v|)>=<frac|P<around|(|v,h|)>|P<around|(|v|)>>=<frac|<frac|1|Z>e<rsup|-E<around|(|v,h|)>>|<frac|1|Z>*<big|sum><rsub|h>e<rsup|-E<around|(|v,h|)>>>=<frac|e<rsup|v<rsup|T>W*h+a<rsup|T>v+b<rsup|T>h>|<big|sum><rsub|h>e<rsup|v<rsup|T>W*h+a<rsup|T>v+b<rsup|T>h>>>

  <math|P<rsub|B*B><around|(|h<rsub|i>=1\|v|)>=<frac|e<rsup|+v<rsup|T>w<rsub|i>1+a<rsup|T>v+b<rsub|i>1>|e<rsup|+v<rsup|T>w<rsub|i>0+a<rsup|T>v+b<rsub|i>0>+e<rsup|+v<rsup|T>w<rsub|i>1+a<rsup|T>v+b<rsub|i>1>>=<frac|e<rsup|+v<rsup|T>w<rsub|i>+b<rsub|i>>|1+e<rsup|+v<rsup|T>w<rsub|i>+b<rsub|i>>>=<frac|1|1+e<rsup|-v<rsup|T>w<rsub|i>-b<rsub|i>>>=sigmoid<around|(|w<rsup|T><rsub|i>v+b<rsub|i>|)>>

  <math|P<rsub|B*B><around*|(|\<b-h\>=\<b-1\><around*|\||\<b-v\>|)>=sigmoid<around*|(|\<b-W\>*\<b-v\>+\<b-b\>|)>|\<nobracket\>>>

  <\math>
    P<rsub|B*B><around*|(|v<rsub|i>=1<around*|\||h|\<nobracket\>>|)>=<frac|e<rsup|+1*w<rsup|<around*|(|T|)>><rsub|i>h+a<rsub|i>1+b<rsup|T>h>|e<rsup|+0*w<rsup|<around*|(|T|)>><rsub|i>h+a<rsub|i>0+b<rsup|T>h>+e<rsup|+1*w<rsup|<around*|(|T|)>><rsub|i>h+a<rsub|i>1+b<rsup|T>h>>=<frac|1|1+e<rsup|-w<rsup|<around*|(|T|)>><rsub|i>h-a<rsub|i>>>=sigmoid<around|(|w<rsup|<around*|(|T|)>><rsub|i>h+a<rsub|i>|)>
  </math>

  <math|P<rsub|B*B><around*|(|\<b-v\>=\<b-1\><around*|\||\<b-h\>|\<nobracket\>>|)>=sigmoid<around*|(|\<b-W\><rsup|T>\<b-h\>+\<b-a\>|)>>

  <strong|Free Energy>

  Next we use the definition of free energy and calculate its derivates on
  parameters <math|W>, <math| a> and <math|b>. Free energy is defined to be:

  <math|F<around|(|v|)>=-log<big|sum><rsub|h>e<rsup|-E<around|(|v,h|)>>>

  And the related probability distribution of data (visible states) is

  <math|P<around|(|v|)>=<frac|1|Z>e<rsup|-F<around|(|v|)>>=<frac|1|Z><big|sum><rsub|h>e<rsup|-E<around|(|v,h|)>>>,
  <math|Z=<big|int>e<rsup|-F<around|(|v|)>>*d*v>

  Now we want to calculate gradient with respect to parameters of the
  distribution in order to maximize likelihood of the data
  <math|P<around|(|v|)>>:

  <math|<frac|\<partial\>-logp<around|(|v|)>|\<partial\>\<theta\>>=-<frac|\<partial\>p<around|(|v|)>/\<partial\>\<theta\>|p<around|(|v|)>>=-<frac|p<around|(|v|)>(-\<partial\>F<around|(|v|)>/\<partial\>\<theta\>)-p<around|(|v|)><around|(|\<partial\>Z/\<partial\>\<theta\>|)>/Z|p<around|(|v|)>>=<frac|\<partial\>F<around|(|v|)>|\<partial\>\<theta\>>+<frac|1|Z>*<frac|\<partial\>Z|\<partial\>\<theta\>>>

  We calculate the term <math|<frac|1|Z>*<frac|\<partial\>Z|\<partial\>\<theta\>>>
  separatedly:

  <math|<frac|1|Z>*<frac|\<partial\>Z|\<partial\>\<theta\>>=<big|int>-<frac|\<partial\>F<around|(|v|)>|\<partial\>\<theta\>><frac|1|Z>e<rsup|-F<around|(|v|)>>d*v=-<big|int>p<around|(|v|)>*<frac|\<partial\>F<around|(|v|)>|\<partial\>\<theta\>>*d*v>

  The general form of the derivate is then:

  <math|<frac|\<partial\>logp<around|(|v|)>|\<partial\>\<theta\>>=-<around*|(|<frac|\<partial\>F<around|(|v|)>|\<partial\>\<theta\>>-<big|int>p<around|(|v|)>*<frac|\<partial\>F<around|(|v|)>|\<partial\>\<theta\>>*d*v|)>=-<around*|(|<frac|\<partial\>F<around|(|v|)>|\<partial\>\<theta\>>-E<rsub|v><around|[|<frac|\<partial\>F<around|(|v|)>|\<partial\>\<theta\>>|]>|)>>

  And the latter term can be approximated using contrastive divergence
  algorithm. We use the training data to produce <math|p<around|(|h\|v|)>>
  and then sample <math|p<around|(|v\|h|)>> and repeat the procedure to get
  sample <math|<around|(|v<rsub|i>,h<rsub|i>|)>> and only keep
  <math|v<rsub|i>> to get the approximate sample from distribution
  <math|p<around|(|v|)>>. This <em|maybe> special case of Gibbs sampling.
  Altenatively, it is possible to use MCMC, AIS, PT-MCMC or some other
  sampling technique to sample directly from <math|P<around*|(|v|)>> instead
  using CD-k Gibbs sampling.\ 

  <strong|Gradient descent>

  Parameters of the distribution are optimized using gradient descent
  algorithm (maximum likelihood method) so it is important to calculate
  actual derivates of <math|p<around|(|v|)>> for Bernoulli-Bernoulli RBM.
  (Also note that gradients are needed by approximative second order methods
  like L-BFGS).

  First we further simplify the <math|F<around|(|v|)>> terms

  <math|F<around|(|v|)>=-log<big|sum><rsub|h>e<rsup|-E<rsub|B*B><around|(|v,h|)>>=-a<rsup|T>v-log<big|sum><rsub|h>e<rsup|<around|(|W<rsup|T>v*+b|)><rsup|T>h>=-a<rsup|T>v-log<big|sum><rsub|h>e<rsup|<big|sum><rsub|i,j>h<rsub|i><around|(|v<rsub|j>*w<rsub|i*j>*+b<rsub|i>|)>*>>

  <math|F<around|(|v|)>=-a<rsup|T>v-log<big|sum><rsub|h><big|prod><rsub|i>e<rsup|*h<rsub|i><around|(|<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>|)>*>=-a<rsup|T>v-<big|sum><rsub|i>log<big|sum><rsub|h<rsub|i>>e<rsup|*h<rsub|i><around|(|<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>|)>*>>

  If we decide <math|h=<around|{|0,1|}>> then the equation simplifies further
  into

  <math|F<around|(|v|)>=-a<rsup|T>v-<big|sum><rsub|i>log<around|(|1+e<rsup|*<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>*>|)>>

  Calculating gradients leads into eqs:

  <math|<frac|\<partial\>F<around|(|v|)>|\<partial\>a>=-v>

  <math|<frac|\<partial\>F<around|(|v|)>|\<partial\>b<rsub|i>>=-<frac|e<rsup|*<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>*><rsup|*>*|1+e<rsup|*<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>*>>=-sigmoid<around|(|<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>|)>>

  <math|<frac|\<partial\>*F<around|(|v|)>|\<partial\>*w<rsub|i*j>>=-<frac|e<rsup|*<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>*>*v<rsub|j>|1+e<rsup|*<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>*>>=-v<rsub|j>*sigmoid<around|(|<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>|)>*>

  <with|font-shape|italic|The calculated gradients can be given to 2nd order
  LBFGS optimizer which uses reconstruction error as the error function and
  proper gradients of free energy <math|F<around*|(|v|)>> as input so we do
  not do simple gradient descent but try to take 2nd order curvature into
  consideration too when minimizing data reconstruction error.>\ 

  \;

  <strong|Gaussian distribution (Gaussian-Bernoulli RBM) - Continuous RBM>

  So far we have only discussed about Bernoulli-Bernoulli RBM. But what we
  really want is to process continuous valued input data (and then maybe use
  BB-RBM to further process its hidden variables). The possible models to use
  are Gaussian-Bernoulli and Beta-Bernoulli (<em|note that Beta and Bernoulli
  distributions are conjugate distributions, this might be useful..>). It
  seems that gaussian distribution is far more popular so I try to calculate
  Gaussian-Bernoulli RBM instead.

  If we ignore (or fix it to unit value) the variance term of normal
  distribution, the logical energy function seem to be:\ 

  <\math>
    E<rsub|G*B><around|(|v,h|)>=<frac|1|2><around|\<\|\|\>|v-a|\<\|\|\>><rsup|2>-v<rsup|T>W*h-b<rsup|T>h
  </math>

  To further justify this model, let's calculate marginalized distributions
  <math|p<around|(|v\|h|)>> and <math|p<around|(|h\|v|)>> for this model.

  <math|P<around|(|v\|h|)>=<frac|P<around|(|v,h|)>|P<around|(|h|)>>=<frac|e<rsup|-E<rsub|G*B><around|(|v,h|)>><rsup|>|<big|int><rsub|v>e<rsup|-E<rsub|G*B><around|(|v,h|)>><rsup|>d*v>=<frac|e<rsup|v<rsup|T>W*h-<frac|1|2><around|\<\|\|\>|v-a|\<\|\|\>><rsup|2>><rsup|>|<big|int><rsub|v>e<rsup|v<rsup|T>W*h-<frac|1|2><around|\<\|\|\>|v-a|\<\|\|\>><rsup|2>>d*v>>

  \;

  <\math>
    <big|int><rsub|v>e<rsup|v<rsup|T>W*h-<frac|1|2><around|\<\|\|\>|v-a|\<\|\|\>><rsup|2>+b<rsup|T>h>d*v=e<rsup|b<rsup|T>h>*<big|int>e<rsup|v<rsup|T>W*h-<frac|1|2><around|\<\|\|\>|v-a|\<\|\|\>><rsup|2>>*d*v=<big|int>e<rsup|-<frac|1|2><around|\<\|\|\>|v|\<\|\|\>><rsup|2>+*v<rsup|T><around|(|W*h*+a|)>-<frac|1|2><around|\<\|\|\>|a|\<\|\|\>><rsup|2>>*d*v
  </math>

  <math|<big|int>e<rsup|-<frac|1|2><around|\<\|\|\>|v|\<\|\|\>><rsup|2>+*v<rsup|T><around|(|W*h*+a|)>-<frac|1|2><around|\<\|\|\>|a|\<\|\|\>><rsup|2>>*d*v=e<rsup|<frac|1|2><around|\<\|\|\>|W*h+a|\<\|\|\>><rsup|2>-<frac|1|2><around|\<\|\|\>|a|\<\|\|\>><rsup|2>><big|int>e<rsup|-<frac|1|2><around|\<\|\|\>|v-<around|(|W*h+a|)>|\<\|\|\>><rsup|2>>*d*v>

  <math|P<around|(|v\|h|)>=<frac|e<rsup|<frac|1|2><around|\<\|\|\>|W*h+a|\<\|\|\>><rsup|2>-<frac|1|2><around|\<\|\|\>|a|\<\|\|\>><rsup|2>>e<rsup|-<frac|1|2><around|\<\|\|\>|v-<around|(|W*h+a|)>|\<\|\|\>><rsup|2>>|e<rsup|<frac|1|2><around|\<\|\|\>|W*h+a|\<\|\|\>><rsup|2>-<frac|1|2><around|\<\|\|\>|a|\<\|\|\>><rsup|2>><big|int>e<rsup|-<frac|1|2><around|\<\|\|\>|v-<around|(|W*h+a|)>|\<\|\|\>><rsup|2>>*d*v>=<frac|1|Z>*e<rsup|-<frac|1|2><around|\<\|\|\>|v-<around|(|W*h+a|)>|\<\|\|\>><rsup|2>>\<sim\>Normal<around|(|W*h+a,I|)>>

  And similar calculations can be done to calculate:
  <math|P<around|(|h\|v|)>=sigmoid<around|(|v<rsup|T>W+b|)>>.

  The related free energy model is:

  <math|F<around|(|v|)>=-log<big|sum><rsub|h>e<rsup|-E<rsub|G*B><around|(|v,h|)>>=-log<big|sum><rsub|h>e<rsup|-<frac|1|2><around|\<\|\|\>|v-a|\<\|\|\>><rsup|2>+v<rsup|T>W*h+b<rsup|T>h>=<frac|1|2><around|\<\|\|\>|v-a|\<\|\|\>><rsup|2>-log<big|sum><rsub|h>e<rsup|<around|(|W<rsup|T>v*+b|)><rsup|T>h>>

  And the related gradient of normal distribution parameter <math|a> is:

  <math|<frac|\<partial\>F|\<partial\>a>=a-v>

  And the other gradients should be the same as in the Bernoulli-Bernoulli
  model.

  \;

  <strong|Non-unit variance>

  Unit variance assumption of Gaussian-Bernoulli RBM can be a problem when
  the input data is not properly preprocessed (in practice we assume
  <math|Var<around|[|f<around|(|x|)>+e|]>=Var<around|[|f<around|(|x|)>|]>+N<around|(|0,1|)>>).
  For example, maximum variance of Bernoulli distributed hidden variables is
  0.25 which means that the noise in input data should be modelled to be much
  smaller.

  <with|font-shape|italic|NOTE: We should add a inverse-Wishart prior for
  <math|\<Sigma\>> term assuming it to be diagnonal <math|I> initially (a
  regularizer). After this we can normalize variance of each dimension to
  have unit variance meaning that prior is hopefully close to noise prior and
  integrate over the <math|p<around*|(|\<Sigma\>|)>> prior.>

  The energy model for the non-unit variance that can make sense is:

  <math|E<rsub|G*B><around|(|v,h|)>=<frac|1|2><around|(|v-a|)><rsup|T>\<Sigma\><rsup|-1><around|(|v-a|)>-(\<Sigma\><rsup|-0.5>v*)<rsup|T>W*h-b<rsup|T>h-log<around*|(|<around*|\||\<Sigma\>|\|><rsup|-<around*|(|v+p+1|)>/2>|)>+<frac|1|2>tr<around*|(|\<Sigma\><rsup|-1>|)>>

  which can be further justified by calculating <math|p<around|(|v\|h|)>> and
  <math|p<around|(|h\|v|)>> distributions:

  <math|P<around|(|v\|h|)>=<frac|P<around|(|v,h|)>|P<around|(|h|)>>=<frac|e<rsup|-E<rsub|G*B><around|(|v,h|)>><rsup|>|<big|int><rsub|v>e<rsup|-E<rsub|G*B><around|(|v,h|)>><rsup|>d*v>=<frac|e<rsup|<around|(|\<Sigma\><rsup|-0.5>*v|)><rsup|T>W*h-<frac|1|2><around|(|v-a|)><rsup|T>\<Sigma\><rsup|-1><around|(|v-a|)>><rsup|>|<big|int><rsub|v>e<rsup|<around|(|\<Sigma\><rsup|-0.5>*v|)><rsup|T>W*h-<frac|1|2><around|(|v-a|)><rsup|T>\<Sigma\><rsup|-1><around|(|v-a|)>>d*v>>

  \;

  <\math>
    <big|int><rsub|v>e<rsup|<around|(|\<Sigma\><rsup|-0.5>*v|)><rsup|T>W*h-<frac|1|2><around|(|v-a|)><rsup|T>\<Sigma\><rsup|-1><around|(|v-a|)>>d*v=*<big|int>e<rsup|-<frac|1|2>v<rsup|T>\<Sigma\><rsup|-1>v+*v<rsup|T>\<Sigma\><rsup|-1><around|(|\<Sigma\><rsup|0.5>W*h*+a|)>-<frac|1|2><around|\<\|\|\>|a|\<\|\|\>><rsup|2>>*d*v
  </math>

  <math|<big|int>e<rsup|-<frac|1|2>v<rsup|T>\<Sigma\><rsup|-1>v+*v<rsup|T>\<Sigma\><rsup|-1><around|(|\<Sigma\><rsup|0.5>W*h*+a|)>-<frac|1|2><around|\<\|\|\>|a|\<\|\|\>><rsup|2>>*d*v=e<rsup|<frac|1|2><around|\<\|\|\>|\<Sigma\><rsup|-0.5>W*h+a|\<\|\|\>><rsub|\<Sigma\><rsup|>><rsup|2>-<frac|1|2><around|\<\|\|\>|a|\<\|\|\>><rsup|2>><big|int>e<rsup|-<frac|1|2><around|\<\|\|\>|v-<around|(|\<Sigma\><rsup|0.5>W*h+a|)>|\<\|\|\>><rsub|\<Sigma\>><rsup|2>>*d*v>

  <math|P<around|(|v\|h|)>=<frac|1|Z<around|(|\<Sigma\>|)>>*e<rsup|-<frac|1|2><around|\<\|\|\>|v-<around|(|\<Sigma\><rsup|0.5>W*h+a|)>|\<\|\|\>><rsub|\<Sigma\>><rsup|2>>\<sim\>Normal<around|(|v\|\<Sigma\><rsup|1/2>*W*h+a,\<Sigma\>|)>>

  <math|P<around*|(|\<b-v\><around*|\||\<b-h\>|\<nobracket\>>,<with|font-series|bold|\<Sigma\>>|)>\<sim\>Normal<around*|(|v<around*|\||\<b-Sigma\><rsup|1/2>*\<b-W\>*\<b-h\>+\<b-a\>,\<b-Sigma\>|\<nobracket\>>|)>>

  \;

  To generate normally distributed variables with wanted covariance matrix we
  notice that for <math|x\<sim\>N<around|(|0,I|)>> data we have:

  <math|\<Sigma\><rsub|x>=Cov<around|[|A*x|]>=E<around|[|A*x*x<rsup|T>*A<rsup|T>|]>=A*I*A<rsup|T>=A*A<rsup|T>=X\<Lambda\>*X<rsup|T>\<Rightarrow\>A=X*\<Lambda\><rsup|1/2>>

  \;

  And similarly we calculate

  <math|P<around|(|h\|v|)>=<frac|P<around|(|v,h|)>|P<around|(|v|)>>=<frac|e<rsup|-E<rsub|G*B><around|(|v,h|)>>|<big|sum><rsub|h>e<rsup|-E<rsub|G*B><around|(|v,h|)>>>=<frac|e<rsup|<around|(|\<Sigma\><rsup|-0.5>*v|)><rsup|T>W*h+b<rsup|T>h><rsup|>|<rsub|><big|sum><rsub|h>e<rsup|<around|(|\<Sigma\><rsup|-0.5>*v|)><rsup|T>W*h+b<rsup|T>h>>=<frac|e<rsup|<around|(|\<Sigma\><rsup|-0.5>*v|)><rsup|T>W*h+b<rsup|T>h><rsup|>|<rsub|><big|sum><rsub|h>e<rsup|<around|(|\<Sigma\><rsup|-0.5>*v|)><rsup|T>W*h+b<rsup|T>h>>>

  The term can be further rewritten as:

  <math|e<rsup|<around|(|\<Sigma\><rsup|-0.5>*v|)><rsup|T>W*h+b<rsup|T>h><rsup|>=e<rsup|<big|sum><rsub|d><around|(|<around|(|\<Sigma\><rsup|-0.5>*v|)><rsup|T>w<rsub|d>**+b*<rsub|d>|)>*h<rsub|d>>=<big|prod><rsub|d><rsup|>e<rsup|<around|(|<around|(|\<Sigma\><rsup|-0.5>*v|)><rsup|T>w<rsub|d>**+b*<rsub|d>|)>*h<rsub|d>><rsup|>>

  And if we only restrict to a single variable we have

  <math|P<around|(|h<rsub|d>\|v|)>=<frac|e<rsup|<around|(|<around|(|\<Sigma\><rsup|-0.5>*v|)><rsup|T>w<rsub|d>**+b*<rsub|d>|)>*h<rsub|d>>|<big|sum><rsub|h<rsub|d>>e<rsup|<around|(|<around|(|\<Sigma\><rsup|-0.5>*v|)><rsup|T>w<rsub|d>**+b*<rsub|d>|)>*h<rsub|d>>>=sigmoid<around|(|<around|(|\<Sigma\><rsup|-0.5>*v|)><rsup|T>w<rsub|d>**+b*<rsub|d>|)>>.

  <math|P<around|(|\<b-h\>\|\<b-v\>,<with|font-series|bold|\<Sigma\>>|)>=sigmoid<around|(|<around|(|\<b-Sigma\><rsup|-0.5>*\<b-v\>|)><rsup|T>\<b-W\>**+\<b-b\>*|)>>

  \;

  \;

  <strong|Free Energy gradient of <math|\<Sigma\>>>

  Next we want to calculate gradient to <math|\<Sigma\>> which can be very
  tricky. In the continuous RBM paper I managed to find, the authors didn't
  use matrix form and only calculated gradient of <math|\<sigma\>> and then
  <em|approximated> it. This result further shows that calculation of gradent
  of full covariance matrix <math|\<Sigma\>> in this case is probably
  analytically extremely difficult or impossible.

  Initially, we notice that <math|\<Sigma\>> is a symmetric matrix. This mean
  that if we do a change of basis we can always find a space where
  <math|\<Sigma\>> is a diagonal matrix, <strong|<math|\<Sigma\>=diag<around|(|\<sigma\><rsub|1>\<ldots\>\<sigma\><rsub|D>|)>>.>
  Additionally, the RBM model implicitly assumes that variables are
  independent when given hidden or visible variables. Therefore, to continue
  to make this independence assumption (note: RBM can be seen as somewhat
  similar to ICA where we estimate ``independent'' or ``semi-independent''
  components from the data), we assume there is no correlations between
  elements of the visible units given hidden units.

  \;

  <\math>
    F<around|(|v|)>=-log<big|sum><rsub|h>e<rsup|-E<rsub|G*B><around|(|v,h|)>>=-log<big|sum><rsub|h>e<rsup|-<frac|1|2><around|\<\|\|\>|v-a|\<\|\|\>><rsub|\<Sigma\>><rsup|2>+<around|(|\<Sigma\>*<rsup|-0.5>v|)><rsup|T>W*h+b<rsup|T>h+log<around*|(|<around*|\||\<Sigma\>|\|><rsup|-<around*|(|v+p+1|)>/2>|)>-<frac|1|2>tr<around*|(|\<Sigma\><rsup|-1>|)>>

    =<frac|1|2><around|(|v-a|)><rsup|T>\<Sigma\><rsup|-1><around|(|v-a|)>-log<around*|(|<around*|\||\<Sigma\>|\|><rsup|-<around*|(|v+p+1|)>/2>|)>+<frac|1|2>tr<around*|(|\<Sigma\><rsup|-1>|)>-log<big|sum><rsub|h>e<rsup|<around|(|W<rsup|T>\<Sigma\><rsup|-0.5>v*+b|)><rsup|T>h>

    =<frac|1|2><around|(|v-a|)><rsup|T>\<Sigma\><rsup|-1><around|(|v-a|)>-log<around*|(|<around*|\||\<Sigma\>|\|><rsup|-<around*|(|v+p+1|)>/2>|)>+<frac|1|2>tr<around*|(|\<Sigma\><rsup|-1>|)>-log<big|prod><rsub|i><big|sum><rsub|h<rsub|i>>e<rsup|<around|(|W<rsup|T>\<Sigma\><rsup|-0.5>v*+b|)><rsub|i>*h<rsub|i>>

    =<frac|1|2><around|(|v-a|)><rsup|T>\<Sigma\><rsup|-1><around|(|v-a|)>+<around*|(|<around*|(|v+p+1|)>/2|)>*log<around*|(|<around*|\||\<Sigma\>|\|>|)>+<frac|1|2><big|sum><rsub|i><frac|1|\<sigma\><rsup|2><rsub|i>>-<big|sum><rsub|i>log<around|(|1+e<rsup|<around|(|W<rsup|T>\<Sigma\><rsup|-0.5>v*+b|)><rsub|i>*>|)>
  </math>

  Its derivate, assuming the covariance matrix is diagonal, is:\ 

  <\math>
    <frac|\<partial\>F|\<partial\>\<sigma\><rsub|k>>=-<frac|<around|(|v<rsub|k>-a<rsub|k>|)><rsup|2>|\<sigma\><rsup|3><rsub|k>>+<around*|(|v+p+1|)><big|sum><rsub|i><frac|\<partial\>|\<partial\>\<sigma\><rsub|k>>log*<around*|(|\<sigma\><rsub|i>|)>-<frac|1|\<sigma\><rsub|k><rsup|3>>-<big|sum><rsub|i><frac|e<rsup|<around|(|W<rsup|T>\<Sigma\><rsup|-0.5>v*+b|)><rsub|i>*>|1+e<rsup|<around|(|W<rsup|T>\<Sigma\><rsup|-0.5>v*+b|)><rsub|i>*>>*<frac|\<partial\>|\<partial\>\<sigma\><rsub|k>><around|(|<big|sum><rsub|k>w<rsub|k*i>*<frac|v<rsub|k>|\<sigma\><rsub|k>>|)>

    =-<frac|<around|(|v<rsub|k>-a<rsub|k>|)><rsup|2>|\<sigma\><rsup|3><rsub|k>>+<around*|(|v+p+1|)><frac|1|\<sigma\><rsub|k>>-<frac|1|\<sigma\><rsub|k><rsup|3>>+v<rsub|k>/\<sigma\><rsup|2><rsub|k>*<big|sum><rsub|i>w<rsub|k*i>*sigmoid<around|(|W<rsup|T>\<Sigma\><rsup|-0.5>v*+b|)><rsub|i>**
  </math>

  <\math>
    <rsup|>diag<around|[|<frac|\<partial\>*F|\<partial\>*\<Sigma\>>|]>=

    diag[-\<Sigma\><rsup|-3/2><around*|(|<around|(|v-a|)>*<around|(|v-a|)><rsup|T>+I|)>+<around*|(|v+p+1|)>\<Sigma\><rsup|-1/2>+*<around|(|\<Sigma\><rsup|-1>v|)>*<around|(|W*sigmoid<around|(|W<rsup|T>\<Sigma\><rsup|-0.5>v+b|)>|)><rsup|T>]
  </math>

  \;

  Other derivates of the free energy are:

  <math|<frac|\<partial\>F|\<partial\>a>=-\<Sigma\><rsup|-1><around|(|v-a|)>>

  <math|<frac|\<partial\>F<around|(|v|)>|\<partial\>b<rsub|i>>=-sigmoid<around|(|<big|sum><rsub|j>w<rsub|i*j>v<rsub|j>*/\<sigma\><rsub|j>*+b<rsub|i>|)>>,
  <math|<frac|\<partial\>*F|\<partial\>b>=-sigmoid<around|(|W<rsup|T>\<Sigma\><rsup|-1/2>v+b|)>>

  <math|<frac|\<partial\>*F<around|(|v|)>|\<partial\>*w<rsub|i*j>>=-<frac|v<rsub|j>|\<sigma\><rsub|j>>*sigmoid<around|(|<big|sum><rsub|j>w<rsub|i*j>**v<rsub|j>/\<sigma\><rsub|j>+b<rsub|i>|)>>,<math|<frac|\<partial\>*F|\<partial\>*W>=-sigmoid<around|(|W<rsup|T>\<Sigma\><rsup|-1/2>v+b|)>*<around|(|\<Sigma\><rsup|-1/2><rsup|>*v|)><rsup|T>>

  \;

  Now the open question is whether these derivates are valid for
  <em|<strong|any>> covariance matrix <math|\<Sigma\>> as it would mean that
  we could then estimate correlations between visible layer inputs and not
  expect them to be statistically independent given hidden variables.

  <em|RESULTS: This does not seem to work for any <math|\<Sigma\>> matrix,
  but it is uncertain as the primary problem here is that variances can
  become negative meaning that the method is not robust or a complex method
  is needed to dynamically adjust learning rates so that <math|\<Sigma\>>
  matrix (eigenvalues) can never become negative.>

  It is possible to get this working but it seems that the convergence is
  really slow. The primary idea is to check if updated covariance matrix has
  positive diagonal and is otherwise good looking (non complex numbers, NaNs,
  Infs etc) and always drop the learning rate by 50% if it seems that the
  update would make covariance matrix ``unhealthy''. But this is not robust
  or a good solution as it ``filters out'' updates that would turn matrix
  into bad one. Additionally, the update rule seem to be really SLOW here.
  (<verbatim|gbrbm-variance-model-v1>)

  The direct variance update don't work, we instead try to fix the situation
  with the change of variable <math|e<rsup|-z<rsub|i>>=1/\<sigma\><rsup|2><rsub|i>>
  (as recommended below), leading into the equation:

  <\math>
    F<around*|(|v|)>=<frac|1|2><around|(|v-a|)><rsup|T>\<Sigma\><rsup|-1><around|(|v-a|)>+<around*|(|<around*|(|v+p+1|)>/2|)>*log<around*|(|<around*|\||\<Sigma\>|\|>|)>+<frac|1|2>tr<around*|(|\<Sigma\><rsup|-1>|)>-<big|sum><rsub|i>log<around|(|1+e<rsup|<around|(|W<rsup|T>\<Sigma\><rsup|-0.5>v*+b|)><rsub|i>*>|)>

    =<frac|1|2><big|sum><rsub|i><around*|(|1+<around*|(|v<rsub|i>-a<rsub|i>|)><rsup|2>*|)>e<rsup|-z<rsub|i>>+<around*|(|<around*|(|v+p+1|)>/2|)><big|sum><rsub|i>z<rsub|i>-<big|sum><rsub|j>log<around|(|1+exp<around*|(|<big|sum><rsub|k>w<rsub|k*j>*v<rsub|k>*e<rsup|-z<rsub|k>/2>+b<rsub|j>|)>|)>
  </math>

  <math|<frac|\<partial\>F|\<partial\>z<rsub|i>>=-<frac|1|2><around*|(|1+<around*|(|v<rsub|i>-a<rsub|i>|)><rsup|<rsup|2>>|)>e<rsup|-z<rsub|i>>+<around*|(|<around*|(|v+p+1|)>/2|)>-><math|<big|sum><rsub|j>sigmoid<around*|(|W<rsup|T>\<Sigma\><rsup|-0.5>v*+b|)><rsub|j><around*|(|-<frac|1|2>w<rsub|i*j>v<rsub|i>*e<rsup|-z<rsub|i>/2>|)>>

  <math|<frac|\<partial\>F|\<partial\>z<rsub|i>>=-<frac|1|2>*e<rsup|-z<rsub|i>><around*|(|1+<around*|(|v<rsub|i>-a<rsub|i>|)><rsup|<rsup|2>>|)>+<around*|(|<around*|(|2*dim<around*|(|v|)>+1|)>/2|)>+<frac|1|2>e<rsup|-z<rsub|i>/2>><math|*v<rsub|i><big|sum><rsub|j>*w<rsub|i*j>*sigmoid<around*|(|W<rsup|T>\<Sigma\><rsup|-0.5>v*+b|)><rsub|j>>

  \;

  Wishart prior's degrees of freedom is chosen to be low so we choose the
  minimum <math|v=p> so that mean of inverse wishart prior/regularizer so
  degrees of freedom is minimized and variance is maximized (maximum
  unceretainty).

  <with|font-shape|italic|NOTE: Using (Wishart) prior to the problem does not
  improve results and can even harm the optimization. <strong|INSTEAD, one
  should do optimization of non-variance terms (a,b,W) and variance terms (z)
  separatedly by going to local optimum and then switching to another set of
  variables and continue doing this until convergence which can typically
  find the underlying real model.>> Alternatively, in sampling one should
  maybe do Gibbs sampling and sample z and then (a,b,W) and then again z and
  then again (a,b,W) using separative sampler for (z) and (a,b,W).

  \;

  This works somewhat better but <with|font-series|bold|still> leads into
  NaNs and other problems when the larger number of hidden nodes are used.
  (This may be because of small covariance matrixes leading into eqs near
  Infinities.)

  \;

  <strong|ALTERNATIVE GAUSSIAN MODEL>

  Because their basic approach to introduce variances into model seem to
  cause problems I now try another slightly modified model introduced by
  Aalto university researchers (Cho et al. 2011).

  Now the energy of the system is defined to be\ 

  <\math>
    E<rsub|G*B><around|(|v,h|)>

    =<frac|1|2>\<alpha\><big|sum><rsub|i><around|(|v<rsub|i>-a<rsub|i>|)><rsup|2>/\<sigma\><rsup|2><rsub|i>-<big|sum><rsub|i,j><frac|v<rsub|i>|\<sigma\><rsup|2><rsub|i>>w<rsub|i*j>h<rsub|j>-<big|sum><rsub|i>b<rsub|i>h<rsub|i>

    =<frac|1|2><around|(|v-a|)><around*|(|\<alpha\>D<rsup|-2>|)><around|(|v-a|)>-v<rsup|T>D<rsup|-2>W*h-b<rsup|T>h

    =<big|sum><rsub|i><frac|1|2>*v<rsup|2><rsub|i>*\<alpha\>/\<sigma\><rsup|2><rsub|i>+<frac|1|2>a<rsup|2><rsub|i>*\<alpha\>/\<sigma\><rsup|2><rsub|i>-<big|sum><rsub|i><around|(|a<rsub|i>+<big|sum><rsub|j>w<rsub|i*j>h<rsub|j>/\<alpha\>|)>v<rsub|i>*\<alpha\>/\<sigma\><rsup|2><rsub|i>-<big|sum><rsub|i>b<rsub|i>h<rsub|i>

    =<frac|1|2><big|sum><rsub|i><around|(|v<rsub|i>-<around|(|a<rsub|i>+<big|sum><rsub|j>w<rsub|i*j>h<rsub|j>/\<alpha\>|)>|)><rsup|2>\<alpha\>/\<sigma\><rsup|2><rsub|i>+<frac|1|2><big|sum><rsub|i>a<rsup|2><rsub|i>\<alpha\>/\<sigma\><rsup|2><rsub|i>-<frac|1|2><big|sum><rsub|i><around|(|a<rsub|i>+<big|sum><rsub|j>w<rsub|i*j>h<rsub|j>/\<alpha\>|)><rsup|2>*\<alpha\>/\<sigma\><rsup|2><rsub|i>-<big|sum><rsub|i>b<rsub|i>h<rsub|i>

    =<frac|1|2><around|(|v-<around|(|a+W*h/\<alpha\>|)>|)><rsup|T><around*|(|\<alpha\>*D<rsup|-2>|)><around|(|v-<around|(|a+W*h/\<alpha\>|)>|)>+<frac|1|2>a<rsup|T><around*|(|\<alpha\>*D<rsup|-2>|)>a-<frac|1|2><around|\<\|\|\>|a+W*h/\<alpha\>|\<\|\|\>><rsub|<around*|(|D<rsup|2>/\<alpha\>|)>><rsup|2>-b<rsup|T>h
  </math>

  After this it is again quite straightforward to calculate the conditional
  probabilities given hidden or visible neurons.

  <\math>
    P<around|(|v\|h|)>=<frac|e<rsup|-<frac|1|2><big|sum><rsub|i><around|(|v<rsub|i>-<around|(|a<rsub|i>+<big|sum><rsub|j>w<rsub|i*j>h<rsub|j>/\<alpha\>|)>|)><rsup|2>\<alpha\>/\<sigma\><rsup|2><rsub|i>><rsup|>|<big|int>e<rsup|-<frac|1|2><big|sum><rsub|i><around|(|v<rsub|i>-<around|(|a<rsub|i>+<big|sum><rsub|j>w<rsub|i*j>h<rsub|j>/\<alpha\>|)>|)><rsup|2>\<alpha\>/\<sigma\><rsup|2><rsub|i>*+\<ldots\>.>*d*v>\<propto\>e<rsup|-<frac|1|2><big|sum><rsub|i><around|(|v<rsub|i>-<around|(|a<rsub|i>+<big|sum><rsub|j>w<rsub|i*j>h<rsub|j>/\<alpha\>|)>|)><rsup|2>\<alpha\>/\<sigma\><rsup|2><rsub|i>>

    \<sim\><big|prod><rsub|i>N<around|(|a<rsub|i>+<big|sum><rsub|j>w<rsub|i*j>h<rsub|j>/\<alpha\>,\<sigma\><rsup|2><rsub|i>/\<alpha\>|)>
  </math>

  <strong|><strong|<math|P<around|(|v\|h|)>\<sim\>Normal<around|(|a+W*h/\<alpha\>,D<rsup|2>/\<alpha\>|)>>>,
  <strong|<strong|<strong|<math|D<rsup|2>=diag<around|[|\<sigma\><rsup|2><rsub|1>\<ldots\>\<sigma\><rsup|2><rsub|D>|]>>>>>

  <strong|<strong|>><strong|>

  <math|P<around|(|h<rsub|j>\|v|)>=<frac|e<rsup|<big|sum><rsub|i,j><frac|v<rsub|i>|\<sigma\><rsup|2><rsub|i>>w<rsub|i*j>h<rsub|j>+<big|sum><rsub|j>b<rsub|j>h<rsub|j>><rsup|><rsup|>|<big|sum><rsub|h<rsub|j>>e<rsup|<big|sum><rsub|i,j><frac|v<rsub|i>|\<sigma\><rsup|2><rsub|i>>w<rsub|i*j>h<rsub|j>+<big|sum><rsub|j>b<rsub|j>h<rsub|j>>>=<frac|e<rsup|<around|(|<big|sum><rsub|i><frac|v<rsub|i>|\<sigma\><rsup|2><rsub|i>>w<rsub|i*j>+b<rsub|j>|)>h<rsub|j>><rsup|><rsup|>|<big|sum><rsub|h<rsub|j>>e<rsup|<around|(|<big|sum><rsub|i><frac|v<rsub|i>|\<sigma\><rsup|2><rsub|i>>w<rsub|i*j>+b<rsub|j>|)>h<rsub|j>><rsup|>>=sigmoid<around|(|<big|sum><rsub|i><frac|v<rsub|i>|\<sigma\><rsup|2><rsub|i>>w<rsub|i*j>+b<rsub|j>|)>>

  <strong|<math|P<around|(|h\|v|)>=sigmoid<around|(|b+v<rsup|T>D<rsup|-2>W|)>>>

  \;

  Notice here that this time the mean is not related to variances which can
  be useful.

  \;

  After calculating the conditional probabilities, we again calculate free
  energy of the energy function (and then calculate its partial derivates).

  <\math>
    F<around|(|v|)>=-log<big|sum><rsub|h>e<rsup|-E<rsub|G*B><around|(|v,h|)>>=-log<big|sum><rsub|h>e<rsup|-<frac|1|2>*\<alpha\>*<around|(|v-a|)><rsup|T>D<rsup|-2><around|(|v-a|)>+v<rsup|T>D<rsup|-2>W*h+b<rsup|T>h>

    =<rsup|><frac|1|2>\<alpha\>*<around|(|v-a|)><rsup|T>D<rsup|-2><around|(|v-a|)>-<big|sum><rsub|i>log<big|sum><rsub|h<rsub|i>>e<rsup|<around|(|W<rsup|T>*D<rsup|-2>**v+b|)><rsub|i>h<rsub|i>>

    =<frac|1|2>\<alpha\>*<around|(|v-a|)><rsup|T>D<rsup|-2><around|(|v-a|)>-<big|sum><rsub|i>log<around|(|1+e<rsup|<around|(|W<rsup|T>*D<rsup|-2>**v+b|)><rsub|i>>|)>

    =<frac|1|2><big|sum><rsub|i>\<alpha\>*<frac|<around|(|v<rsub|i>-a<rsub|i>|)><rsup|2>|\<sigma\><rsup|2><rsub|i>>-<big|sum><rsub|i>log<around|(|1+exp<around|(|<big|sum><rsub|j>w<rsub|j*i>v<rsub|j>/\<sigma\><rsup|2><rsub|j>+b<rsub|i>|)>|)>
  </math>

  \;

  From this formulation we can calculate

  <\math>
    <frac|\<partial\>F|\<partial\>a>=-\<alpha\>*D<rsup|-2><around|(|a-v|)>
  </math>

  <math|<frac|\<partial\>F|\<partial\>b<rsub|i>>=-<frac|e<rsup|<around|(|W<rsup|T>*D<rsup|-2>**v+b|)><rsub|i>>|1+e<rsup|<around|(|W<rsup|T>*D<rsup|-2>**v+b|)><rsub|i>>>=-sigmoid<around|(|W<rsup|T>D<rsup|-2>v+b|)><rsub|i>>

  <math|<frac|\<partial\>F|\<partial\>b>=-sigmoid<around|(|W<rsup|T>D<rsup|-2>v+b|)>=-h>

  <math|<frac|\<partial\>F|\<partial\>w<rsub|i*j>>=-<frac|e<rsup|<around|(|W<rsup|T>*D<rsup|-2>**v+b|)><rsub|i>>|1+e<rsup|<around|(|W<rsup|T>*D<rsup|-2>**v+b|)><rsub|i>>>*(*v<rsub|j>/\<sigma\><rsup|2><rsub|j>)=-<frac|v<rsub|j>|\<sigma\><rsub|j><rsup|2>>*sigmoid<around|(|W<rsup|T>D<rsup|-2>v+b|)><rsub|i>=-D<rsup|<rsup|-2>>v*h<rsup|<rsup|T>>>

  \;

  \;

  But the crucial derivate is the derivate of <math|\<sigma\><rsub|i>> terms,
  we alter energy function by the change of terms:
  <math|1/\<sigma\><rsup|2><rsub|i>=e<rsup|-z<rsub|i>>> leading into formula:

  <math|F<around|(|v|)>=<frac|1|2><big|sum><rsub|i>\<alpha\>*<around|(|v<rsub|i>-a<rsub|i>|)><rsup|2>*e<rsup|-z<rsub|i>>-<big|sum><rsub|j>log<around|(|1+exp<around|(|<big|sum><rsub|i>w<rsub|i*j>v<rsub|i>*e<rsup|-z<rsub|i>>+b<rsub|j>|)>|)>>

  And then derivate with respect to <math|z<rsub|i>>:

  <\math>
    <frac|\<partial\>F|\<partial\>z<rsub|i>>=-<frac|1|2>\<alpha\>*<around|(|v<rsub|i>-a<rsub|i>|)><rsup|2>*e<rsup|-z<rsub|i>>+<big|sum><rsub|j><frac|exp<around|(|<big|sum><rsub|i>w<rsub|i*j>v<rsub|i>*e<rsup|-z<rsub|i>>+b<rsub|j>|)>|1+exp<around|(|<big|sum><rsub|i>w<rsub|i*j>v<rsub|i>*e<rsup|-z<rsub|i>>+b<rsub|j>|)>>*w<rsub|i*j>*v<rsub|i>*e<rsup|-z<rsub|i>>

    =-e<rsup|-z<rsub|i>>*<around|[|<frac|1|2>\<alpha\>*<around|(|v<rsub|i>-a<rsub|i>|)><rsup|2>*-v<rsub|i><big|sum><rsub|j>w*<rsub|i*j>sigmoid<around|(|W<rsup|T>D<rsup|-2>v+b|)><rsub|j>|]>=e<rsup|-z<rsub|i>>*S<around*|(|a,b,D|)>
  </math>

  \;

  But because variance is difficult parameter to optimize for, we want to
  calculate its Hessian matrix and use second order derivation

  \;

  \;

  This is the exactly same formula as the one given in the paper of Aalto uni
  people (<em|Improved Learning of Gaussian-Bernoulli Restricted Boltzmann
  Machines. ICANN 2011.>)

  <with|font-series|bold|Important>: The analysis of results seem to imply to
  following result. Aalto university people model performance with data
  having multiple modes (circle data etc) have <with|font-series|bold|bad
  performance> if gradient descent method is used and variance should be
  learned - cannot learn the variance (Parallel Tempering MCMC may be
  possible to learn the model variance correctly). However, it has
  <with|font-series|bold|BETTER> performance if correct variance is known and
  gradient descent method is used.

  <\enumerate>
    <item>This seem to imply to following approach: first optimize
    ``original'' GB-RBM so that you can learn the variances (correctly)

    <item>After learning the variances switch to ``Aalto-university'' GB-RBM
    model which can give better reconstruction error. With correct variances
    known it is now possible to learn multiple modes of the distribution
    easily.

    <item>Alternatively try to use more sophisticated methods.
  </enumerate>

  \;

  <strong|Parallel Tempering Annihilated Importance Sampling (AIS)>

  Parallel tempering is a powerful approach that was used by Aalto university
  people in 2011 (also see <with|font-shape|italic|On the Quantitative
  Analysis of Deep Belief Networks>. <with|font-shape|italic|Salakhutdinov.>
  about AIS). I will try to use same method to generate intermediate
  distributions from which to estimate partition function
  <math|<with|font-series|bold|Z>> which is needed to estimate exact
  probabilities.

  <center|<math|P<around*|(|v|)>=<with|font-series|bold|<frac|1|Z>><big|sum><rsub|h>e<rsup|-E<around*|(|v,h|)>>>>

  We generate series of temperatures <math|\<beta\>=<around*|[|0\<ldots\>1|]>>
  and define intermediate models as (<with|font-shape|italic|Cho et al.
  2011>):

  <\center>
    <math|\<b-W\><rsup|<around*|(|\<beta\>|)>>=\<beta\>*\<b-W\>> ,
    <math|a<rsub|i>=\<beta\>*a<rsub|i>+<around*|(|1-\<beta\>|)>*m<rsub|i>>

    <math|\<b-b\><rsup|<around*|(|\<beta\>|)>>=*\<beta\>*\<b-b\>>,
    <math|\<sigma\><rsub|i><rsup|<around*|(|\<beta\>|)>>=<sqrt|\<beta\>*\<sigma\><rsup|2><rsub|i>+<around*|(|1-\<beta\>|)>*s<rsup|2><rsub|i>>>
  </center>

  \;

  So with <math|\<beta\>=0> we simply sample from unscaled normal probability
  distribution <math|v<rsub|i>\<sim\>\<cal-N\><around*|(|m<rsub|i>,s<rsup|2><rsub|i>|)>>
  and then define <with|font-series|bold|stochastic> transition operation as\ 

  <center|<math|T<rsub|\<beta\>><around*|(|v<rprime|'>,v|)>=p<rsub|\<beta\>><around*|(|v<rprime|'><around*|\||h|\<nobracket\>>|)>*p<rsub|\<beta\>><around*|(|h<around*|\||v|\<nobracket\>>|)>>>

  Because we know (<with|font-series|bold|proof?>) that iterative Gibbs
  sampling <math|T<around*|(|v<rprime|'>,v|)>> from distribution will lead to
  distribution <math|p<around*|(|v|)>> given any distribution
  <math|p<around*|(|v|)>> for any <math|v> (assuming all
  <math|p<around*|(|v|)> \<gtr\> 0>) then the transition from probability
  distribution <math|p<around*|(|v|)>> of values will lead to limiting
  distribution of <math|p<around*|(|v|)>> of values?? (Convergence point of
  markov chain).

  After this we use AIS to estimate <math|Z<rsub|>>-ratios (which mean is
  used as an estimate) using un-scaled distributions by using samples:

  <center|<math|<frac|Z<rsub|\<beta\><rsub|K>>|Z<rsub|\<beta\><rsub|0>>>=<frac|p<rprime|'><rsub|\<beta\><rsub|1>><around*|(|\<b-v\><rsub|1>|)>|p<rprime|'><rsub|\<beta\><rsub|0>><around*|(|\<b-v\><rsub|1>|)>>*<frac|p<rprime|'><rsub|\<beta\><rsub|2>><around*|(|\<b-v\><rsub|2>|)>|p<rprime|'><rsub|\<beta\><rsub|1>><around*|(|\<b-v\><rsub|2>|)>>\<ldots\><frac|p<rprime|'><rsub|\<beta\><rsub|K>><around*|(|\<b-v\><rsub|K>|)>|p<rprime|'><rsub|\<beta\><rsub|k-1>><around*|(|\<b-v\><rsub|K>|)>>**>.>

  (NOTE: is <math|\<b-v\><rsub|K>> generated this way also distributed as
  <math|p<rsub|K><around*|(|\<b-v\>|)>> ?)

  But because we know <math|p<rsub|\<beta\><rsub|0>><around*|(|\<b-v\>|)>> to
  be exactly normal it is trival to sample from and its normalizing constant
  <math|Z> can be computed directly.

  <center|<math|Z<rsub|\<beta\><rsub|0>>=<around*|(|2*\<pi\>|)><rsup|D/2>*\<sigma\><rsub|1>\<ldots\>\<sigma\><rsub|D>>>

  What is still needed to estimate <math|Z> is the calculation of un-scaled
  probability distribution values <math|p<rprime|'><around*|(|\<b-v\>|)>>
  given parameters of GB-RBM. This needs summation over all values of
  <math|\<b-h\>>:

  <center|<\math>
    P<rprime|'><around*|(|\<b-v\>|)>=<with|font-series|bold|>e<rsup|-<frac|1|2><around*|(|\<b-v\>-\<b-a\>|)><rsup|T>\<b-Sigma\><rsup|-1><around*|(|\<b-v\>-\<b-a\>|)>>*<big|sum><rsub|\<b-h\>>e<rsup|\<b-v\><rsup|T>\<b-Sigma\><rsup|-0.5>\<b-W\>*h+\<b-b\><rsup|T>\<b-h\>>
  </math>>

  For <math|\<b-h\>> this simplifies further into summation:

  <center|<math|S=<big|sum><rsub|\<b-h\>>e<rsup|\<b-alpha\><rsup|T>\<b-h\>>=<big|sum><rsub|\<b-h\>><big|prod><rsub|i>e<rsup|\<alpha\><rsub|i>*h<rsub|i>>>>

  The next comes the tricky/smart step, because <em|<strong|h>> takes only
  values <math|0> and <math|1> this summation can be simplified without going
  through all the <math|2<rsup|dim<around*|(|\<b-h\>|)>>> possible values by
  using property of multiplication <math|<around*|(|\<alpha\>+\<beta\>|)><around*|(|\<varsigma\>+\<delta\>|)>\<ldots\>.>
  ``generates''/''goes through'' all possible <math|2<rsup|D>> states
  <with|font-series|bold|in linear time><with|font-series|bold|!>

  <center|<math|S=<big|sum><rsub|\<b-h\>><big|prod><rsub|i>e<rsup|\<alpha\><rsub|i>*h<rsub|i>>=<big|prod><rsub|i><around*|(|e<rsup|\<alpha\><rsub|i>*<around*|(|h<rsub|i>=0|)>>+e<rsup|\<alpha\><rsub|i>*<around*|(|h<rsub|i>=1|)>>|)>=<big|prod><rsub|i><around*|(|1+e<rsup|\<alpha\><rsub|i>*>|)>>>

  Therefore, the unscaled log probability is:

  <\center>
    <math|log<around*|(|P<rprime|'><around*|(|\<b-v\>|)>|)>=-<frac|1|2><around*|(|\<b-v\>-\<b-a\>|)><rsup|T>\<b-Sigma\><rsup|-1><around*|(|\<b-v\>-\<b-a\>|)>+<big|sum><rsub|i>log<around*|(|1+e<rsup|\<alpha\><rsub|i><around*|(|\<b-v\>,\<b-W\>,\<b-Sigma\>,\<b-b\>|)>>|)>>,

    <math|\<b-alpha\>=\<b-W\><rsup|T>\<b-Sigma\><rsup|-0.5>*\<b-v\>+\<b-b\>>.
  </center>

  <with|font-series|bold|About numerical stability: >in practice, the
  exponent of alpha can become too large. If <math|e<rsup|\<alpha\><rsub|i>>>
  becomes floating point infinity and destroys numerical accuracy, we use
  approximation\ 

  <center|<math|log<around*|(|1+e<rsup|\<alpha\>>|)>\<thickapprox\>log<around*|(|e<rsup|\<alpha\>>|)>=\<alpha\>>>

  which is good approaximation if <math|e<rsup|\<alpha\>>> is near infinity
  (even at floating point accuracy).

  \;

  <strong|Building Neural Network from Stacked RBM>

  After the problem of learning GB-RBM and BB-RBM has been solved it is
  naturally interesting to try to use it to construct
  <with|font-shape|italic|feedforward neural network>. This is rather
  straight forward as the probability functions always has sigmoidal
  probability function (even at the top layer) and we can directly insert
  weights and biases from the RBM. The only problem is that our RBM is a
  stochastic/probabilistic meaning that we will now make significant error
  when approximating hidden neurons to have continuos values.

  <with|font-series|bold|TODO>

  \;

  <strong|Hamiltonian Monte Carlo sampler based approach>

  Because the direct optimization method doesn't seem to work very well. It
  seems that an interesting approach could be try to use Monte Carlo sampling
  as seen in many papers as the given probability model fits into HMC
  approach. For hamiltonian we will use function

  <\center>
    <math|H<around*|(|\<b-q\>,\<b-p\>|)>=E<rsub|T><around*|(|\<b-q\>|)>+K<around*|(|\<b-q\>|)>>,
    <math|K<around*|(|\<b-p\>|)>=\<b-p\><rsup|T>\<b-M\><rsup|<rsup|-1>>\<b-p\>/2>
  </center>

  And we want to use HMC to sample from distribution:

  <center|<math|p<around*|(|\<b-theta\><around*|\||\<b-v\>|\<nobracket\>>|)>=<frac|1|Z<around*|(|\<b-v\>|)>>*exp<around*|(|-E<rsub|T><around*|(|\<b-theta\><around*|\||\<b-v\>|\<nobracket\>>|)>|)>\<propto\>p<around*|(|\<b-v\><around*|\|||\<nobracket\>>\<b-theta\>|)>*p<around*|(|\<b-theta\>|)>>>

  But in practice we have only know the data likelihood
  <math|p<around*|(|\<b-v\><around*|\||\<b-theta\>|\<nobracket\>>|)>=<frac|1|Z<around*|(|\<b-theta\>|)>>e<rsup|-F<around*|(|\<b-v\><around*|\||\<b-theta\>|\<nobracket\>>|)>>>
  term from the previous sections. This will lead to the following eqs (for
  the gradient):

  <\center>
    <math|E<rsub|T><around*|(|\<b-theta\><around*|\||\<b-v\>|\<nobracket\>>|)>=-log<around*|(|<frac|Z<around*|(|\<b-v\>|)>|Z<rsub|T><around*|(|\<b-theta\>|)>>|)>+F<rsub|T><around*|(|\<b-v\><around*|\||\<b-theta\>|\<nobracket\>>|)>-log<around*|(|<frac|p<around*|(|\<b-theta\>|)>|p<around*|(|\<b-v\>|)>>|)>
    >

    <math|\<nabla\><rsub|\<b-theta\>>E<rsub|T><around*|(|\<b-theta\><around*|\||\<b-v\>|\<nobracket\>>|)>=\<nabla\><rsub|\<b-theta\>>F<rsub|T><around*|(|\<b-v\><around*|\||\<b-theta\>|\<nobracket\>>|)>-*E<rsub|\<b-v\>><around*|[|\<nabla\><rsub|\<b-theta\>>F<rsub|T><around*|(|\<b-v\><around*|\||\<b-theta\>|\<nobracket\>>|)>|]>-\<nabla\><rsub|\<b-theta\>>log<around*|(|p<around*|(|\<b-theta\>|)>|)>
    >
  </center>

  We can decide for the flat priors meaning that
  <math|p<around*|(|\<b-theta\>|)>\<propto\>1> and the last term will be
  zero. This means that we can use directly gradient computed for the
  <math|-log<around*|(|P<around*|(|\<b-v\>|)>|)>>. But HMC also calculates
  difference between energy functions

  <center|<math|E<rsub|T><around*|(|\<b-theta\><rsub|n+1><around*|\||\<b-v\>|\<nobracket\>>|)>-E<rsub|T><around*|(|\<b-theta\><rsub|n><around*|\||\<b-v\>|\<nobracket\>>|)>=F<rsub|T><around*|(|\<b-v\><around*|\||\<b-theta\><rsub|n+1>|\<nobracket\>>|)>-F<rsub|T><around*|(|\<b-v\><around*|\||\<b-theta\><rsub|n>|\<nobracket\>>|)>+log<around*|(|<frac|Z<rsub|T><around*|(|\<b-theta\><rsub|n+1>|)>|Z<rsub|T><around*|(|\<b-theta\><rsub|n>|)>>|)>-log<around*|(|<frac|p<around*|(|\<b-theta\><rsub|n+1>|)>|p<around*|(|\<b-theta\><rsub|n>|)>>|)>>>

  The last term can be assumed to be zero but the problem is ratio of
  partition functions. We know that ratio of partition functions will be
  close to <math|1> if the change between parameters is not too large so it
  can be initially approximated to be zero but in practice this seem to lead
  to too small variances causing problems. But we can approximate by using
  the previous results (by using first order derivate approximation from
  <math|\<b-theta\><rsub|n+1>> to <math|\<b-theta\><rsub|n>> and from
  <math|\<b-theta\><rsub|n>> to <math|\<b-theta\><rsub|n+1>> and calculating
  the mean):

  <center|<math|log<around*|(|<frac|Z<rsub|T><around*|(|\<b-theta\><rsub|n+1>|)>|Z<rsub|T><around*|(|\<b-theta\><rsub|n>|)>>|)>\<thickapprox\>-<frac|1|2><around*|(|\<b-theta\><rsub|n+1>-\<b-theta\><rsub|n>|)><rsup|T>*<around*|(|E<rsub|\<b-v\>><around*|[|\<nabla\><rsub|\<b-theta\>>F<rsub|T><around*|(|\<b-v\><around*|\||\<b-theta\>|\<nobracket\>><rsub|n+1>|)>|]>+E<rsub|\<b-v\>><around*|[|\<nabla\><rsub|\<b-theta\>>F<rsub|T><around*|(|\<b-v\><around*|\||\<b-theta\>|\<nobracket\>><rsub|n>|)>|]>|)>>>

  \;

  <with|font-series|bold|This approximation does not seem to work well in
  practice and cannot be used.>

  You can calculate gradient quite easily but you cannot calculate difference
  between energy functions as it requires estimating the ratio of partition
  functions. For this case I will try to use <em|importance sampling> (as the
  distributions should be close to each other and we want only the ratio and
  not the actual <math|Z<rsub|T>>-values. Theory:

  <center|<math|<frac|Z<rsub|T><around*|(|\<b-theta\><rsub|n+1>|)>|Z<rsub|T><around*|(|\<b-theta\><rsub|n>|)>>=<big|int><frac|p<rprime|'><rsub|T><around*|(|\<b-v\><around*|\|||\<nobracket\>>\<b-theta\><rsub|n+1>|)>|Z<rsub|T><around*|(|\<b-theta\><rsub|n>|)>>d\<b-v\>=<big|int><frac|p<rprime|'><rsub|T><around*|(|\<b-v\><around*|\|||\<nobracket\>>\<b-theta\><rsub|n+1>|)>|p<rsub|><rsub|n><rprime|'><around*|(|\<b-v\><around*|\|||\<nobracket\>>\<b-theta\><rsub|n>|)>>*p<around*|(|\<b-v\><around*|\|||\<nobracket\>>\<b-theta\><rsub|n>|)>d\<b-v\>=E<rsub|\<b-theta\><rsub|n>><around*|[|<frac|p<rprime|'><rsub|T><around*|(|\<b-v\><around*|\|||\<nobracket\>>\<b-theta\><rsub|n+1>|)>|p<rsub|><rsub|n><rprime|'><around*|(|\<b-v\><around*|\|||\<nobracket\>>\<b-theta\><rsub|n>|)>>|]>>>

  So we need to sample <math|\<b-v\>>:s from the distribution using AIS and
  ``parallel tempering'' approach described by Cho et. al. in 2011. This seem
  to work rather well after which we calculate ratios of unscaled
  distributions <math|p<rprime|'><around*|(|\<b-v\><around*|\|||\<nobracket\>>\<b-theta\>|)>=e<rsup|-F<around*|(|\<b-v\><around*|\||\<b-theta\>|\<nobracket\>>|)>>>
  and calculate the mean value: \ 

  <center|<math|E<rsub|\<b-v\><around*|\|||\<nobracket\>>\<b-theta\><rsub|n>><around*|[|<frac|p<rprime|'><rsub|T><around*|(|\<b-v\><around*|\|||\<nobracket\>>\<b-theta\><rsub|n+1>|)>|p<rsub|><rsub|n><rprime|'><around*|(|\<b-v\><around*|\|||\<nobracket\>>\<b-theta\><rsub|n>|)>>|]>=E<rsub|\<b-v\><around*|\|||\<nobracket\>>\<b-theta\><rsub|n>><around*|[|e<rsup|F<around*|(|\<b-v\><around*|\||\<b-theta\><rsub|n>|\<nobracket\>>|)>-F<around*|(|\<b-v\><around*|\||\<b-theta\><rsub|n+1>|\<nobracket\>>|)>>|]>>.>

  This is easy to parallelize and seem to work with enough accuracy,
  <with|font-series|bold|in low dimensions> (\<less\>256). TODO: Probably a
  good way to estimate <math|Z<rsub|T><around*|(|\<b-theta\><rsub|n>|)>>
  would be to use variational methods (variatioanl bayes but currently it is
  too complicated for me).

  \;

  <with|font-series|bold|Latent variable model>\ 

  In general, the RBM method works by first defining probability function
  <math|p<around*|(|\<b-v\>,\<b-h\><around*|\||\<b-theta\>|)>|\<nobracket\>>>
  and then maximizing visible states (visible data) probability

  <center|<math|max<rsub|\<b-theta\>>*p<around*|(|\<b-v\><around*|\||\<b-theta\>|\<nobracket\>>|)>=<big|int>p<around*|(|\<b-v\>,\<b-h\><around*|\||\<b-theta\>|\<nobracket\>>|)>*d\<b-h\>>>

  But instead of calculating maximum likelihood solution, one could also
  calculate bayesian samples and estimate distribution or sample from\ 

  <center|<math|p<around*|(|\<b-theta\><around*|\||\<b-v\>|\<nobracket\>>|)>\<propto\>p<around*|(|\<b-v\><around*|\||\<b-theta\>|\<nobracket\>>|)>*p<around*|(|\<b-theta\>|)>>>

  But sampling from this distribution can be difficult but one can maybe use
  MCMC sampling which is only interested in relative probabilities

  <center|<math|r=<frac|p<around*|(|\<b-theta\><rsub|<rsub|n+1>><around*|\||\<b-v\>|\<nobracket\>>|)>*p<around*|(|\<b-theta\>|)>|p<around*|(|\<b-theta\><rsub|n><around*|\||\<b-v\>|\<nobracket\>>|)>*p<around*|(|\<b-theta\>|)>>=<frac|<big|sum><rsub|\<b-h\>>p<around*|(|\<b-v\>,\<b-h\><around*|\||\<b-theta\><rsub|n+1>|\<nobracket\>>|)>|<big|sum><rsub|\<b-h\>>p<around*|(|\<b-v\>,\<b-h\><around*|\||\<b-theta\><rsub|n>|\<nobracket\>>|)>>>>

  If we furthermore assume that <math|Z<around*|(|\<b-theta\><rsub|n+1>|)>\<thickapprox\>Z<around*|(|\<b-theta\><rsub|n>|)>>
  then we get a formula:

  <center|<math|r<around*|(|v|)>\<thickapprox\><frac|<big|sum><rsub|\<b-h\>>e<rsup|-E<around*|(|\<b-v\>,\<b-h\><around*|\||\<b-theta\><rsub|n+1>|\<nobracket\>>|)>>|<big|sum><rsub|\<b-h\>>e<rsup|-E<around*|(|\<b-v\>,\<b-h\><around*|\||\<b-theta\><rsub|n>|\<nobracket\>>|)>>>\<thickapprox\>>>

  \;

  \;

  After which hidden states are sampled or calculated using a formula

  <center|<math|p<around*|(|\<b-h\><around*|\||\<b-v\>|\<nobracket\>>|)>=<big|int>p<around*|(|\<b-h\><around*|\||\<b-theta\>,\<b-v\>|\<nobracket\>>|)>*p<around*|(|\<b-theta\><around*|\||\<b-v\>|\<nobracket\>>|)>*d\<b-theta\>\<thickapprox\><frac|1|N><big|sum><rsub|i>p<around*|(|\<b-h\><around*|\||\<b-theta\><rsub|i>,\<b-v\>|\<nobracket\>>|)>>>

  Now for multivariate probability distribution
  <math|\<b-x\>=<around*|[|\<b-v\>,\<b-h\>|]><rsup|T>>,
  <math|p<around*|(|\<b-x\>|)>\<sim\>N<around*|(|\<b-mu\><rsub|\<b-x\>>,\<b-Sigma\><rsub|\<b-x\>>|)>>
  it is trivially easy to compute <math|p<around*|(|\<b-v\><around*|\||\<b-mu\><rsub|\<b-v\>>,\<b-Sigma\><rsub|\<b-v\>>|\<nobracket\>>|)>>
  as the solution is yet another normal distribution. But now because
  marginal distribution do not have dependency to
  <math|\<b-mu\><rsub|\<b-h\>>> or <math|\<b-Sigma\><rsub|\<b-h\>>> or other
  variables given known variable, it is not possible to calculate relatioship
  between <math|\<b-h\>> and <strong|v>.

  If one does not want to use bayesian approach when maximizing probability
  of visible states given parameters, another approach could be still use
  gradient descent but now giving the gradient to L-BFGS optimizer method
  (conjugate gradient) method which should then be able to do higher quality
  maximization of visible states probability by using only gradient
  information.

  \;

  Solving variance model with bayesian prior distribution

  We add inverse scaled chi-squared distribution as prior/regularizer
  distribution for <math|\<sigma\><rsup|2><rsub|i>> terms in order to keep
  variance as low as possible

  \;

  <\math>
    E<rsub|G*B><around|(|v,h|)>+ScaledInv\<chi\><rsup|2><around*|(|\<b-sigma\><rsup|2>|)>

    =<frac|1|2><big|sum><rsub|i><around|(|v<rsub|i>-a<rsub|i>|)><rsup|2>/\<sigma\><rsup|2><rsub|i>-<big|sum><rsub|i,j><frac|v<rsub|i>|\<sigma\><rsup|2><rsub|i>>w<rsub|i*j>h<rsub|j>-<big|sum><rsub|i>b<rsub|i>h<rsub|i>+<frac|1|2><around*|(|\<alpha\>-1|)>*<big|sum><rsub|i><around*|(|v<rsub|i>-a<rsub|i>|)><rsup|<rsup|2>>/\<sigma\><rsup|2><rsub|i>

    \;

    =<frac|1|2>\<alpha\>*<around|(|v-<around|(|a+W*h|)>|)><rsup|T>D<rsup|-2><around|(|v-<around|(|a+W*h|)>|)>+<frac|1|2>a<rsup|T>D<rsup|-2>a-<frac|1|2><around|\<\|\|\>|a+W*h|\<\|\|\>><rsup|2>-b<rsup|T>h
  </math>

  After this It is possible to calculate free energy derivate of
  <math|\<sigma\><rsub|i><rsup|2>>.

  <\math>
    F<around|(|v|)>=-log<big|sum><rsub|h>e<rsup|-E<rsub|G*B><around|(|v,h|)>>=-log<big|sum><rsub|h>e<rsup|-<frac|1|2><around|(|v-a|)><rsup|T>D<rsup|-2><around|(|v-a|)>-\<alpha\><frac|1|2>s<rsup|T>*D*s+v<rsup|T>D<rsup|-2>W*h+b<rsup|T>h>

    =<rsup|><frac|1|2><around|(|v-a|)><rsup|T>D<rsup|-2><around|(|v-a|)>+<frac|1|2>s<rsup|T>D<rsup|2>s-<big|sum><rsub|i>log<big|sum><rsub|h<rsub|i>>e<rsup|<around|(|W<rsup|T>*D<rsup|-2>**v+b|)><rsub|i>h<rsub|i>>

    =<frac|1|2><around|(|v-a|)><rsup|T>D<rsup|-2><around|(|v-a|)>+<frac|1|2>s<rsup|T>D<rsup|-2>s-<big|sum><rsub|i>log<around|(|1+e<rsup|<around|(|W<rsup|T>*D<rsup|-2>**v+b|)><rsub|i>>|)>

    =<frac|1|2><big|sum><rsub|i><frac|<around|(|v<rsub|i>-a<rsub|i>|)><rsup|2>|\<sigma\><rsup|2><rsub|i>>+<frac|1|2><big|sum><rsub|i>s<rsup|2><rsub|i>\<sigma\><rsup|2><rsub|i>-<big|sum><rsub|i>log<around|(|1+exp<around|(|<big|sum><rsub|j>w<rsub|j*i>v<rsub|j>/\<sigma\><rsup|2><rsub|j>+b<rsub|i>|)>|)>
  </math>

  \;

  And the derivate with respect to <math|z<rsub|i>> is:

  <\math>
    <frac|\<partial\>F|\<partial\>z<rsub|i>>=-e<rsup|-z<rsub|i>>*<around|[|<frac|1|2><around|(|v<rsub|i>-a<rsub|i>|)><rsup|2>*-v<rsub|i><big|sum><rsub|j>w*<rsub|i*j>sigmoid<around|(|W<rsup|T>D<rsup|-2>v+b|)><rsub|j>|]>+<frac|1|2>s<rsup|2><rsub|i>e<rsup|z<rsub|i>>
  </math>

  And <math|s<rsub|i>> is a sample square statistic
  <math|s<rsup|2><rsub|i>=<big|sum><rsup|N><rsub|j=1><around*|(|v<rsub|i><around*|(|j|)>-<wide|v<rsub|i>|\<bar\>><around*|(|j|)>|)><rsup|2>>
  where <math|j>:s are indexes for individual observations from the data. Now
  from the data it is easy to calculate <math|s<rsup|2><rsub|i>> but from the
  model this is a different thing. If we approximate
  <math|s<rsup|2><rsub|i>=\<alpha\>*<around*|(|v<rsub|i>-a<rsub|i>|)><rsup|2>>
  we get regularized gradients.

  \;

  <with|font-series|bold|Multiple visible elements>

  In real world applications, it is often interesting to learn distribution
  <math|p<around*|(|\<b-y\><around*|\||\<b-x\>|\<nobracket\>>|)>> and in RBM
  context this means we have multiple different visible elements
  <math|\<b-v\>=<around*|[|\<b-v\><rsub|1>,\<b-v\><rsub|2>|]>> and we need to
  be able to sample from <math|p<around*|(|\<b-v\><rsub|2><around*|\||\<b-v\><rsub|1>|\<nobracket\>>|)>>
  in order to predict value of <math|\<b-v\><rsub|2>> given elements
  <math|\<b-v\><rsub|1>>. In this model, the learning of RBM elements can be
  done normally. However, in reconstruction phase we need to sample from
  <math|p<around*|(|\<b-h\><around*|\||\<b-v\><rsub|1>|\<nobracket\>>|)>> and
  then from <math|p<around*|(|\<b-v\><rsub|2><around*|\||\<b-h\>|\<nobracket\>>|)>>
  in order to approximate sample from integral

  <center|<math|p<around*|(|\<b-v\><rsub|2><around*|\||\<b-v\><rsub|1>|\<nobracket\>>|)>=<big|int>p<around*|(|\<b-v\><rsub|2><around*|\||\<b-h\>|\<nobracket\>>|)>*p<around*|(|\<b-h\><around*|\||\<b-v\><rsub|1>|\<nobracket\>>|)>*d\<b-h\>>>

  In GB-RBM we need that <math|p<around*|(|\<b-v\><around*|\||\<b-h\>|\<nobracket\>>|)>>
  has a normal distribution and calculating conditional normal distribution
  <math|p<around*|(|\<b-v\><rsub|1><around*|\||\<b-h\>|\<nobracket\>>|)>> and
  <math|p<around*|(|\<b-v\><rsub|2><around*|\||\<b-h\>|\<nobracket\>>|)>> is
  ``easy'':

  <\center>
    <math|p<around*|(|\<b-v\><around*|\||\<b-h\>|\<nobracket\>>|)>\<sim\>Normal<around*|(|\<b-v\><around*|\||\<b-Sigma\><rsup|1/2>*\<b-W\>*\<b-h\>+\<b-a\>,\<b-Sigma\>|\<nobracket\>>|)>,\<b-Sigma\>=diag<around*|(|<around*|[|\<sigma\><rsub|1>\<ldots\>\<sigma\><rsub|D>|]>|)>>

    <math|p<around*|(|\<b-v\><rsub|2><around*|\||\<b-v\><rsub|1>=\<b-x\>|\<nobracket\>>,\<b-h\>|)>\<sim\>Normal<around*|(|<around*|\<nobracket\>|\<b-v\><rsub|2><around*|\||\<b-mu\><rsub|2>+\<b-Sigma\><rsub|21>*\<b-Sigma\><rsup|-1><rsub|11>|(>\<b-x\>-\<b-mu\><rsub|1>|)>,\<b-Sigma\><rsub|22>-\<b-Sigma\><rsub|21>*\<b-Sigma\><rsup|-1><rsub|11>*\<b-Sigma\><rsub|12>|)>>
  </center>

  But because <math|\<b-Sigma\>> is diagonal conditional distribution does
  not directly depend on values of <math|\<b-v\><rsub|1>>.

  <\center>
    <math|p<around*|(|\<b-v\><rsub|1><around*|\||\<b-h\>|\<nobracket\>>|)>\<sim\>Normal<around*|(|\<b-v\><rsub|1><around*|\||\<b-mu\><rsub|1><around*|(|\<b-h\>|)>,\<b-Sigma\><rsub|11>|\<nobracket\>>|)>>

    <math|p<around*|(|\<b-v\><rsub|2><around*|\||\<b-h\>|)>\<sim\>Normal<around*|(|\<b-v\><rsub|2><around*|\||\<b-mu\><rsub|2>|(>\<b-h\>|)>,\<b-Sigma\><rsub|22>|)>>
  </center>

  What is left is to calculate <math|p<around*|(|\<b-h\><around*|\||\<b-v\><rsub|1>|\<nobracket\>>|)>>
  to generate hidden variables from input values
  <math|\<b-v\><rsub|1>=\<b-x\>>, which is more difficult.

  <center|<math|p<around|(|\<b-h\>\|\<b-v\>|)>=sigmoid<around|(|<around|(|\<b-Sigma\><rsup|-0.5>*\<b-v\>|)><rsup|T>\<b-W\>**+\<b-b\>*|)>>>

  By generating <math|\<b-v\><rsub|2>>:s randomly we can generate
  candidate(s) <math|\<b-h\><rsub|0>> which can be then improved and we can
  use bayes rule to write

  <center|<math|p<around*|(|\<b-h\><around*|\||\<b-v\><rsub|1>|\<nobracket\>>|)>\<propto\>p<around*|(|\<b-v\><rsub|1><around*|\||\<b-h\>|\<nobracket\>>|)>*p<around*|(|\<b-h\>|)>>>

  And if we assume each configuration is is equally likely in our model
  (without any data hidden states should have equal probability of being
  either 0 or 1 <math|\<Rightarrow\>> so the prior should maybe still have
  some effect - it should force values of <math|\<b-h\>> to be either 0 or 1
  thought the values should be equally likely), then the prior
  <math|p<around*|(|\<b-h\>|)>> is flat constant. This makes sampling from
  <math|p<around*|(|\<b-h\><around*|\||\<b-v\><rsub|1>|\<nobracket\>>|)>>
  using HMC rather easy. Our <math|U<around*|(|\<b-h\>|)>> terms are
  (<math|\<b-Gamma\><rsub|1>>-matrix is simple used to select
  <math|\<b-mu\><rsub|1>> from <math|\<b-mu\>>):

  <\center>
    <math|U<around*|(|\<b-h\>|)>=<frac|1|2><around*|(|\<b-v\><rsub|1>-\<b-mu\><rsub|1><around*|(|\<b-h\>|)>|)><rsup|T>\<b-Sigma\><rsup|-1><rsub|11><around*|(|\<b-v\><rsub|1>-\<b-mu\><rsub|1><around*|(|\<b-h\>|)>|)>>,
    <math|\<b-mu\><rsub|1><around*|(|\<b-h\>|)>=\<b-Gamma\><rsub|1><around*|(|\<b-Sigma\><rsup|1/2>*\<b-W\>*\<b-h\>+\<b-a\>|)>>

    <math|\<nabla\><rsub|\<b-h\>>U<around*|(|\<b-h\>|)>=<around*|(|\<b-mu\><rsub|1><around*|(|\<b-h\>|)>-\<b-v\><rsub|1>|)><rsup|T>\<b-Sigma\><rsup|-1><rsub|11><with|font-series|bold|>\<b-Gamma\><rsub|1>*\<b-Sigma\><rsup|1/2>*\<b-W\>*>
  </center>

  Here we have assumed <math|\<b-h\>> can take continuous values between
  <math|0> and <math|1> in practice, the sampling should maybe happen in
  continuous <math|\<b-h\>>-space but the exact samples generated should be
  discretized. Alternatives:

  <\itemize-dot>
    <item>keep <math|\<b-h\>> values discretized to 0, 1 values always. Now
    there is problem that with too small epsilon the sampler stays in a same
    state and cannot never try to change state

    <item>keep values of <math|\<b-h\>> continuous internally and only
    discretize to 0, 1 states when emitting a sample

    <item>use continuous <math|\<b-h\>> values, everywhere, this is not
    well-defined as everything in our theory assumes samples must be
    discretized and other sources recommend against completely continuous
    variables..
  </itemize-dot>

  Another approach would generate candidate <math|\<b-h\><rsub|0>> states and
  then do gradient descent until convergence to the best state and discretize
  <math|\<b-h\>>. After discretization calculate gradient descent again until
  convegence and discretize again <math|\<b-h\>>. Stop until there is no
  changes in <math|\<b-h\>>. Repeat <math|N> times. This will be maximum data
  likelihood estimate of <math|p<around*|(|\<b-h\><around*|\||\<b-v\><rsub|1>|\<nobracket\>>|)>>.
  After this sample <math|\<b-p\><around*|(|\<b-v\><rsub|1><around*|\||\<b-h\>|\<nobracket\>>|)>>.

  <with|font-series|bold|Brute force (modular) approach>

  Assume we can divide <math|\<b-h\><rsub|>> into parts
  <math|<around*|[|\<b-h\><rsub|1>\<ldots\>\<b-h\><rsub|L>|]>> and go through
  all the combinations separatedly. If we have perfect parallel machine this
  happens in <math|O<around*|(|2<rsup|max<rsub|i>*dim<around*|(|\<b-h\><rsub|i>|)>>|)>>
  time or in <math|O<around*|(|2<rsup|dim<around*|(|\<b-h\>|)>/K>|)>> on
  average if we can divide problem perfectly into <math|K >parts. We can
  choose, for example, <math|dim<around*|(|\<b-h\><rsub|i>|)>=10> and the
  computations scale now linearly (although the results are not equally
  good).

  <with|font-series|bold|General multiple elements (modular) approach>

  In practice, the memory requirements <math|O<around*|(|dim<around*|(|\<b-v\>|)>\<times\>dim<around*|(|\<b-h\>|)>|)>>
  and computation of huge RBM's grow too quickly if we use perfect model and
  use full matrix <math|\<b-W\>>. Because of this we consider modular
  approach (also suggested by neuroscience research) where vectors are
  divided into modules/subvectors. This leads into following energy
  equations:

  <\center>
    <\math>
      E<rsub|B*B><around|(|\<b-v\><rsub|1>\<ldots\>\<b-v\><rsub|K>,\<b-h\><rsub|1>\<ldots\>\<b-h\><rsub|L>|)>=-\<b-a\><rsup|T>\<b-v\>-<big|sum><rsub|k,l>\<b-v\><rsub|k><rsup|T>\<b-W\><rsub|k*l>*\<b-h\><rsub|l>-\<b-b\><rsup|T>\<b-h\>

      E<rsub|G*B><around|(|\<b-v\><rsub|1>\<ldots\>\<b-v\><rsub|K>,\<b-h\><rsub|1>\<ldots\>\<b-h\><rsub|L>|)>=<frac|1|2><around|\<\|\|\>|\<b-v\>-\<b-a\>|\<\|\|\>><rsup|2>-<big|sum><rsub|k,l>\<b-v\><rsub|k><rsup|T>\<b-W\><rsub|k*l>*\<b-h\><rsub|l>-\<b-b\><rsup|T>\<b-h\>
    </math>
  </center>

  This reduces amount of memory required during computations but doesn't lead
  into other improvements. But we want that
  <math|p<around*|(|\<b-h\>|)>=p<around*|(|\<b-h\><rsub|1>|)>p<around*|(|\<b-h\><rsub|2>|)>\<ldots\>p<around*|(|\<b-h\><rsub|L>|)>>
  in order to be able to brute force through <math|\<b-h\>> when looking for
  optimum because now we can look for optimum for each <math|\<b-h\><rsub|i>>
  vector separatedly.

  \;

  <emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash><emdash>--

  <strong|Linear optimization>

  Global optimum solution to linear optimization problem\ 

  <center|<math|min e<around*|(|\<b-A\>,\<b-b\>|)>=
  E<rsub|\<b-up-x\>*\<b-up-y\>><around*|{|<frac|1|2><around*|\<\|\|\>|\<b-A\>*\<b-x\>+\<b-b\>-\<b-y\>|\<\|\|\>><rsup|2>|}>>>

  is rather straightforward to calculate and is part of solving many more
  complicated problems, by taking gradients of
  <math|e<around*|(|\<b-A\>,\<b-b\>|)>> and setting them to zero. We get
  equations\ 

  <math|\<b-A\>E<rsub|\<b-x\>><around*|{|\<b-x\>\<b-x\><rsup|T>|}>+\<b-b\>E<rsub|\<b-x\>><around*|{|\<b-x\>|}><rsup|T>*=E<rsub|\<b-x\>*\<b-y\>><around*|{|\<b-y\>*\<b-x\><rsup|T>|}>>

  <math|\<b-A\>E<rsub|\<b-x\>><around*|{|\<b-x\>|}>+\<b-b\>=E<rsub|\<b-y\>><around*|{|\<b-y\>|}>>

  which further simplify to

  <math|\<b-A\><around*|(|E<rsub|\<b-x\>><around*|{|\<b-x\>\<b-x\><rsup|T>|}>-E<rsub|\<b-x\>><around*|{|\<b-x\>|}>E<rsub|\<b-x\>><around*|{|\<b-x\>|}><rsup|T>|)>*=E<rsub|\<b-x\>*\<b-y\>><around*|{|\<b-y\>*\<b-x\><rsup|T>|}>-E<rsub|\<b-y\>><around*|{|\<b-y\>|}>E<rsub|\<b-x\>><around*|{|\<b-x\>|}><rsup|T>>

  <math|\<b-A\>*=\<b-Sigma\><rsub|\<b-y\>*\<b-x\>>\<b-Sigma\><rsup|-1><rsub|\<b-x\>\<b-x\>>>

  and solving <math|\<b-b\>> is also straighforward\ 

  <math|\<b-b\>=E<rsub|\<b-y\>><around*|{|\<b-y\>|}>-\<b-A\>E<rsub|\<b-x\>><around*|{|\<b-x\>|}>>

  \;
</body>

<\initial>
  <\collection>
    <associate|par-hyphen|normal>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
  </collection>
</references>