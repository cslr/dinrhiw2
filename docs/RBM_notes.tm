<TeXmacs|1.0.7.2>

<style|generic>

<\body>
  <strong|Some notes RBM implementation>

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

  <with|font-series|bold|Energy based model of RBM and Continuous RBM (CRBM)>

  The continuous RBM is more complicated to implement and understand. It
  seems that Gaussian-Bernoulli RBM could be the right choice (or maybe
  Beta-Bernoulli RBM) to model continuous input values. Because of this, I
  try to derive the whole BB-RBM and then continuous GB-RBM theory from
  energy based models (EBMs). It is important(?) to remember that
  Bernoulli-Bernoulli RBM model is related to Ising models and statistical
  physics but hidden variable models in general are not.

  The energy of the restricted boltzman machine is:\ 

  <\math>
    E<rsub|B*B>(v,b)=-v<rsup|T>W*h-a<rsup|T>v-b<rsup|T>h

    E<rsub|G*B>(v,h)=-v<rsup|T>W*h-<frac|1|2>\<\|\|\>v-a\<\|\|\><rsup|2>-b<rsup|T>h
  </math>

  Note that we could have added additional constant <math|c> term to these
  equations but it is not needed because probability distributions are
  normalized to have unit probability mass.

  Now the probability of the <math|(v,h)> and only observed variables
  <math|v> is:

  <math|P(v,h)=<frac|1|Z>*e<rsup|-E(v,h)>>,
  <math|Z=<big|sum><rsub|v,h>e<rsup|-E(v,h)>>

  <math|P(v)=<frac|1|Z><big|sum><rsub|h>e<rsup|-E(v,h)>>

  Now the observed variables are Bernoulli distributed, that is, they take
  only values <math|0> and <math|1> which is a strong regularizer fro the
  system (Gaussian-Gaussian RBM is unlikely to work equally well).

  Now we want to calculate probabilities of hidden <math|h> and visible
  <math|v> neurons given probabilities:\ 

  <math|P<rsub|B*B>(h\|v)=<frac|P(v,h)|P(v)>=<frac|<frac|1|Z>e<rsup|-E(v,h)>|<frac|1|Z>*<big|sum><rsub|h>e<rsup|-E(v,h)>>=<frac|e<rsup|v<rsup|T>W*h+a<rsup|T>v+b<rsup|T>h>|<big|sum><rsub|h>e<rsup|v<rsup|T>W*h+a<rsup|T>v+b<rsup|T>h>x>>

  <math|P<rsub|B*B>(h<rsub|i>=1\|v)=<frac|e<rsup|+v<rsup|T>w<rsub|i>1+a<rsup|T>v+b<rsub|i>1>|e<rsup|+v<rsup|T>w<rsub|i>0+a<rsup|T>v+b<rsub|i>0>+e<rsup|+v<rsup|T>w<rsub|i>1+a<rsup|T>v+b<rsub|i>1>>=<frac|e<rsup|+v<rsup|T>w<rsub|i>+b<rsub|i>>|1+e<rsup|+v<rsup|T>w<rsub|i>+b<rsub|i>>>=<frac|1|1+e<rsup|-v<rsup|T>w<rsub|i>-b<rsub|i>>>=sigmoid(w<rsup|T><rsub|i>v+b<rsub|i>)>

  And due to symmetry, a similar calculation can be used to calculate
  <math|P<rsub|B*B>(v<rsub|i>=1\|h)> (for Bernoulli-Bernoulli RBMs).

  <strong|Free Energy>

  Next we use the definition of free energy and calculate its derivates on
  parameters <math|W>, <math| a> and <math|b>. Free energy is defined to be:

  <math|F(v)=-log<big|sum><rsub|h>e<rsup|-E(v,h)>>

  And the related probability distribution of data (visible states) is

  <math|P(v)=<frac|1|Z>e<rsup|-F(v)>=<frac|1|Z><big|sum><rsub|h>e<rsup|-E(v,h)>>,
  <math|Z=<big|int>e<rsup|-F(v)>*d*v>

  Now we want to calculate gradient with respect to parameters of the
  distribution in order to maximize likelihood of the data <math|P(v)>:

  <math|<frac|\<partial\>-logp(v)|\<partial\>\<theta\>>=-<frac|\<partial\>p(v)/\<partial\>\<theta\>|p(v)>=-<frac|p(v)(-\<partial\>F(v)/\<partial\>\<theta\>)-p(v)(\<partial\>Z/\<partial\>\<theta\>)/Z|p(v)>=<frac|\<partial\>F(v)|\<partial\>\<theta\>>+<frac|1|Z>*<frac|\<partial\>Z|\<partial\>\<theta\>>>

  We calculate the term <math|<frac|1|Z>*<frac|\<partial\>Z|\<partial\>\<theta\>>>
  separatedly:

  <math|><math|<frac|1|Z>*<frac|\<partial\>Z|\<partial\>\<theta\>>=<big|int>-<frac|\<partial\>F(v)|\<partial\>\<theta\>><frac|1|Z>e<rsup|-F(v)>d*v=-<big|int>p(v)*<frac|\<partial\>F(v)|\<partial\>\<theta\>>d*v>

  The general form of the derivate is then:

  <with|mode|math|<frac|\<partial\>-logp(v)|\<partial\>\<theta\>>=<frac|\<partial\>F(v)|\<partial\>\<theta\>>-<big|int>p(v)*<frac|\<partial\>F(v)|\<partial\>\<theta\>>*d*v=<frac|\<partial\>F(v)|\<partial\>\<theta\>>-E<rsub|v>[<frac|\<partial\>F(v)|\<partial\>\<theta\>>]>

  And the latter term can be approximated using contrastive divergence
  algorithm. We use the training data to produce <math|p(h\|v)> and then
  sample <math|p(v\|h)> and repeat the procedure to get sample
  <math|(v<rsub|i>,h<rsub|i>)> and only keep <math|v<rsub|i>> to get the
  approximate sample from distribution <math|p(v)>. This <em|maybe> special
  case of Gibbs sampling.

  <strong|Gradient descent>

  Parameters of the distribution are optimized using gradient descent
  algorithm so it is important to calculate actual derivates of <math|p(v)>
  for Bernoulli-Bernoulli RBM.

  First we further simplify the <math|F(v)> term

  <math|F(v)=-log<big|sum><rsub|h>e<rsup|-E<rsub|B*B>(v,h)>=-a<rsup|T>v-log<big|sum><rsub|h>e<rsup|(W<rsup|T>v*+b)<rsup|T>h>=-a<rsup|T>v-log<big|sum><rsub|h>e<rsup|<big|sum><rsub|i,j>h<rsub|i>(v<rsub|j>*w<rsub|i*j>*+b<rsub|i>)*>>

  <math|F(v)=-a<rsup|T>v-log<big|sum><rsub|h><big|prod><rsub|i>e<rsup|*h<rsub|i>(<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>)*>=-a<rsup|T>v-<big|sum><rsub|i>log<big|sum><rsub|h<rsub|i>>e<rsup|*h<rsub|i>(<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>)*>>

  If we decide <math|h={0,1}> then the equation simplifies further into

  <math|F(v)=-a<rsup|T>v-<big|sum><rsub|i>log(1+e<rsup|*<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>*>)>

  Calculating gradients leads into eqs:

  <math|<frac|\<partial\>F(v)|\<partial\>a>=-v>

  <math|<frac|\<partial\>F(v)|\<partial\>b<rsub|i>>=-<frac|e<rsup|*<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>*><rsup|*>*|1+e<rsup|*<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>*>>=-sigmoid(<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>)>

  <math|<frac|\<partial\>*F(v)|\<partial\>*w<rsub|i*j>>=-<frac|e<rsup|*<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>*>*v<rsub|j>|1+e<rsup|*<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>*>>=-v<rsub|j>*sigmoid(<big|sum><rsub|j>v<rsub|j>*w<rsub|i*j>*+b<rsub|i>)*>

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

  <\with|mode|math>
    E<rsub|G*B>(v,h)=<frac|1|2>\<\|\|\>v-a\<\|\|\><rsup|2>-v<rsup|T>W*h-b<rsup|T>h
  </with>

  To further justify this model, let's calculate marginalized distributions
  <math|p(v\|h)> and <math|p(h\|v)> for this model.

  <math|P(v\|h)=<frac|P(v,h)|P(h)>=<frac|e<rsup|-E<rsub|G*B>(v,h)><rsup|>|<big|int><rsub|v>e<rsup|-E<rsub|G*B>(v,h)><rsup|>d*v>=<frac|e<rsup|v<rsup|T>W*h-<frac|1|2>\<\|\|\>v-a\<\|\|\><rsup|2>><rsup|>|<big|int><rsub|v>e<rsup|v<rsup|T>W*h-<frac|1|2>\<\|\|\>v-a\<\|\|\><rsup|2>>d*v>>

  \;

  <\with|mode|math>
    <big|int><rsub|v>e<rsup|v<rsup|T>W*h-<frac|1|2>\<\|\|\>v-a\<\|\|\><rsup|2>+b<rsup|T>h>d*v=e<rsup|b<rsup|T>h>*<big|int>e<rsup|v<rsup|T>W*h-<frac|1|2>\<\|\|\>v-a\<\|\|\><rsup|2>>*d*v=<big|int>e<rsup|-<frac|1|2>\<\|\|\>v\<\|\|\><rsup|2>+*v<rsup|T>(W*h*+a)-<frac|1|2>\<\|\|\>a\<\|\|\><rsup|2>>*d*v
  </with>

  <math|<big|int>e<rsup|-<frac|1|2>\<\|\|\>v\<\|\|\><rsup|2>+*v<rsup|T>(W*h*+a)-<frac|1|2>\<\|\|\>a\<\|\|\><rsup|2>>*d*v=e<rsup|<frac|1|2>\<\|\|\>W*h+a\<\|\|\><rsup|2>-<frac|1|2>\<\|\|\>a\<\|\|\><rsup|2>><big|int>e<rsup|-<frac|1|2>\<\|\|\>v-(W*h+a)\<\|\|\><rsup|2>>*d*v>

  <math|P(v\|h)=<frac|e<rsup|<frac|1|2>\<\|\|\>W*h+a\<\|\|\><rsup|2>-<frac|1|2>\<\|\|\>a\<\|\|\><rsup|2>>e<rsup|-<frac|1|2>\<\|\|\>v-(W*h+a)\<\|\|\><rsup|2>>|e<rsup|<frac|1|2>\<\|\|\>W*h+a\<\|\|\><rsup|2>-<frac|1|2>\<\|\|\>a\<\|\|\><rsup|2>><big|int>e<rsup|-<frac|1|2>\<\|\|\>v-(W*h+a)\<\|\|\><rsup|2>>*d*v>=<frac|1|Z>*e<rsup|-<frac|1|2>\<\|\|\>v-(W*h+a)\<\|\|\><rsup|2>>\<sim\>Normal(W*h+a,I)>

  And similar calculations can be done to calculate:
  <math|P(h\|v)=sigmoid(v<rsup|T>W+b)>.

  The related free energy model is:

  <with|mode|math|F(v)=-log<big|sum><rsub|h>e<rsup|-E<rsub|G*B>(v,h)>=-log<big|sum><rsub|h>e<rsup|-<frac|1|2>\<\|\|\>v-a\<\|\|\><rsup|2>+v<rsup|T>W*h+b<rsup|T>h>=<frac|1|2>\<\|\|\>v-a\<\|\|\><rsup|2>-log<big|sum><rsub|h>e<rsup|(W<rsup|T>v*+b)<rsup|T>h>>

  And the related gradient of normal distribution parameter <math|a> is:

  <math|<frac|\<partial\>F|\<partial\>a>=a-v>

  And the other gradients should be the same as in the Bernoulli-Bernoulli
  model.

  \;

  <strong|Non-unit variance>

  Unit variance assumption of Gaussian-Bernoulli RBM can be a problem when
  the input data is not properly preprocessed (in practice we assume
  <math|Var[f(x)+e]=Var[f(x)]+N(0,1)>). For example, maximum variance of
  Bernoulli distributed hidden variables is 0.25 which means that the noise
  in input data should be modelled to be much smaller.

  The energy model for the non-unit variance that can make sense is:

  <math|E<rsub|G*B>(v,h)=<frac|1|2>(v-a)<rsup|T>\<Sigma\><rsup|-1>(v-a)-(\<Sigma\><rsup|-0.5>v*)<rsup|T>W*h-b<rsup|T>h>

  which can be further justified by calculating <math|p(v\|h)> and
  <math|p(h\|v)> distributions:

  <math|P(v\|h)=<frac|P(v,h)|P(h)>=<frac|e<rsup|-E<rsub|G*B>(v,h)><rsup|>|<big|int><rsub|v>e<rsup|-E<rsub|G*B>(v,h)><rsup|>d*v>=<frac|e<rsup|(\<Sigma\><rsup|-0.5>*v)<rsup|T>W*h-<frac|1|2>(v-a)<rsup|T>\<Sigma\><rsup|-1>(v-a)><rsup|>|<big|int><rsub|v>e<rsup|(\<Sigma\><rsup|-0.5>*v)<rsup|T>W*h-<frac|1|2>(v-a)<rsup|T>\<Sigma\><rsup|-1>(v-a)>d*v>>

  \;

  <\with|mode|math>
    <big|int><rsub|v>e<rsup|(\<Sigma\><rsup|-0.5>*v)<rsup|T>W*h-<frac|1|2>(v-a)<rsup|T>\<Sigma\><rsup|-1>(v-a)>d*v=*<big|int>e<rsup|-<frac|1|2>v<rsup|T>\<Sigma\><rsup|-1>v+*v<rsup|T>\<Sigma\><rsup|-1>(\<Sigma\><rsup|0.5>W*h*+a)-<frac|1|2>\<\|\|\>a\<\|\|\><rsup|2>>*d*v
  </with>

  <math|<big|int>e<rsup|-<frac|1|2>v<rsup|T>\<Sigma\><rsup|-1>v+*v<rsup|T>\<Sigma\><rsup|-1>(\<Sigma\><rsup|0.5>W*h*+a)-<frac|1|2>\<\|\|\>a\<\|\|\><rsup|2>>*d*v=e<rsup|<frac|1|2>\<\|\|\>\<Sigma\><rsup|-0.5>W*h+a\<\|\|\><rsub|\<Sigma\><rsup|>><rsup|2>-<frac|1|2>\<\|\|\>a\<\|\|\><rsup|2>><big|int>e<rsup|-<frac|1|2>\<\|\|\>v-(\<Sigma\><rsup|0.5>W*h+a)\<\|\|\><rsub|\<Sigma\>><rsup|2>>*d*v>

  <math|P(v\|h)==<frac|1|Z(\<Sigma\>)>*e<rsup|-<frac|1|2>\<\|\|\>v-(\<Sigma\><rsup|0.5>W*h+a)\<\|\|\><rsub|\<Sigma\>><rsup|2>>\<sim\>Normal(v\|\<Sigma\><rsup|1/2>*W*h+a,\<Sigma\>)>

  \;

  To generate normally distributed variables with wanted covariance matrix we
  notice that for <math|x\<sim\>N(0,I)> data we have:

  <math|\<Sigma\><rsub|x>=Cov[A*x]=E[A*x*x<rsup|T>*A<rsup|T>]=A*I*A<rsup|T>=A*A<rsup|T>=X\<Lambda\>*X<rsup|T>\<Rightarrow\>A=X*\<Lambda\><rsup|1/2>>

  \;

  And similarly we calculate

  <math|P(h\|v)=<frac|P(v,h)|P(v)>=<frac|e<rsup|-E<rsub|G*B>(v,h)>|<big|sum><rsub|h>e<rsup|-E<rsub|G*B>(v,h)>>=<with|mode|text|<math|<frac|e<rsup|(\<Sigma\><rsup|-0.5>*v)<rsup|T>W*h+b<rsup|T>h><rsup|>|<rsub|><big|sum><rsub|h>e<rsup|(\<Sigma\><rsup|-0.5>*v)<rsup|T>W*h+b<rsup|T>h>>>>=<with|mode|text|<math|<frac|e<rsup|(\<Sigma\><rsup|-0.5>*v)<rsup|T>W*h+b<rsup|T>h><rsup|>|<rsub|><big|sum><rsub|h>e<rsup|(\<Sigma\><rsup|-0.5>*v)<rsup|T>W*h+b<rsup|T>h>>>>>

  The term can be further rewritten as:

  <with|mode|math|e<rsup|(\<Sigma\><rsup|-0.5>*v)<rsup|T>W*h+b<rsup|T>h><rsup|>=e<rsup|<big|sum><rsub|d>((\<Sigma\><rsup|-0.5>*v)<rsup|T>w<rsub|d>**+b*<rsub|d>)*h<rsub|d>>=<big|prod><rsub|d><rsup|>e<rsup|((\<Sigma\><rsup|-0.5>*v)<rsup|T>w<rsub|d>**+b*<rsub|d>)*h<rsub|d>><rsup|>>

  And if we only restrict to a single variable we have

  <math|P(h<rsub|d>\|v)=<frac|e<rsup|((\<Sigma\><rsup|-0.5>*v)<rsup|T>w<rsub|d>**+b*<rsub|d>)*h<rsub|d>>|<big|sum><rsub|h<rsub|d>>e<rsup|((\<Sigma\><rsup|-0.5>*v)<rsup|T>w<rsub|d>**+b*<rsub|d>)*h<rsub|d>>>=sigmoid((\<Sigma\><rsup|-0.5>*v)<rsup|T>w<rsub|d>**+b*<rsub|d>)>.

  <math|P(h\|v)=sigmoid((\<Sigma\><rsup|-0.5>*v)<rsup|T>W**+b*)>

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
  <math|\<Sigma\>> is a diagonal matrix, <strong|<math|\<Sigma\>=diag(\<sigma\><rsub|1>\<ldots\>\<sigma\><rsub|D>)>.>
  Additionally, the RBM model implicitly assumes that variables are
  independent when given hidden or visible variables. Therefore, to continue
  to make this independence assumption (note: RBM can be seen as somewhat
  similar to ICA where we estimate ``independent'' or ``semi-independent''
  components from the data), we assume there is no correlations between
  elements of the visible units given hidden units.

  \;

  <\with|mode|math>
    F(v)=-log<big|sum><rsub|h>e<rsup|-E<rsub|G*B>(v,h)>=-log<big|sum><rsub|h>e<rsup|-<frac|1|2>\<\|\|\>v-a\<\|\|\><rsub|\<Sigma\>><rsup|2>+(\<Sigma\>*<rsup|-0.5>v)<rsup|T>W*h+b<rsup|T>h>

    =<frac|1|2>(v-a)<rsup|T>\<Sigma\><rsup|-1>(v-a)-log<big|sum><rsub|h>e<rsup|(W<rsup|T>\<Sigma\><rsup|-0.5>v*+b)<rsup|T>h>

    =<frac|1|2>(v-a)<rsup|T>\<Sigma\><rsup|-1>(v-a)-log<big|prod><rsub|i><big|sum><rsub|h<rsub|i>>e<rsup|(W<rsup|T>\<Sigma\><rsup|-0.5>v*+b)<rsub|i>*h<rsub|i>>

    =<frac|1|2>(v-a)<rsup|T>\<Sigma\><rsup|-1>(v-a)-<big|sum><rsub|i>log(1+e<rsup|(W<rsup|T>\<Sigma\><rsup|-0.5>v*+b)<rsub|i>*>)
  </with>

  Its derivate, assuming the covariance matrix is diagonal, is:\ 

  <\math>
    <frac|\<partial\>F|\<partial\>\<sigma\><rsub|k>>=-<frac|(v<rsub|k>-a<rsub|k>)<rsup|2>|\<sigma\><rsup|3><rsub|k>>-<big|sum><rsub|i><frac|e<rsup|(W<rsup|T>\<Sigma\><rsup|-0.5>v*+b)<rsub|i>*>|1+e<rsup|(W<rsup|T>\<Sigma\><rsup|-0.5>v*+b)<rsub|i>*>>*<frac|\<partial\>|\<partial\>\<sigma\><rsub|k>>(<big|sum><rsub|k>w<rsub|k*i>*<frac|v<rsub|k>|\<sigma\><rsub|k>>)

    =-<frac|(v<rsub|k>-a<rsub|k>)<rsup|2>|\<sigma\><rsup|3><rsub|k>>+v<rsub|k>/\<sigma\><rsup|2><rsub|k><big|sum><rsub|i>w<rsub|k*i>*sigmoid(W<rsup|T>\<Sigma\><rsup|-0.5>v*+b)<rsub|i>**
  </math>

  <math|<rsup|<with|font-base-size|11|<with|font-base-size|12|>>>diag[<frac|\<partial\>*F|\<partial\>*\<Sigma\>>]=diag[-\<Sigma\><rsup|-3/2>(v-a)*(v-a)<rsup|T>+*(\<Sigma\><rsup|-1>v)*(W*sigmoid(W<rsup|T>\<Sigma\><rsup|-0.5>v+b))<rsup|T>]>

  \;

  Other derivates of the free energy are:

  <math|<frac|\<partial\>F|\<partial\>a>=\<Sigma\><rsup|-1>(a-v)>

  <math|<frac|\<partial\>F(v)|\<partial\>b<rsub|i>>=-sigmoid(<big|sum><rsub|j>w<rsub|i*j>v<rsub|j>*/\<sigma\><rsub|j>*+b<rsub|i>)>,
  <math|<frac|\<partial\>*F|\<partial\>b>=-sigmoid(W<rsup|T>\<Sigma\><rsup|-1/2>v+b)>

  <math|<frac|\<partial\>*F(v)|\<partial\>*w<rsub|i*j>>=-<frac|v<rsub|j>|\<sigma\><rsub|j>>*sigmoid(<big|sum><rsub|j>w<rsub|i*j>**v<rsub|j>/\<sigma\><rsub|j>+b<rsub|i>)>,<with|mode|math|<frac|\<partial\>*F|\<partial\>*W>=-sigmoid(W<rsup|T>\<Sigma\><rsup|-1/2>v+b)*(\<Sigma\><rsup|-1/2><rsup|>*v)<rsup|T>>

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

  \;

  <strong|ALTERNATIVE GAUSSIAN MODEL>

  Because the basic approach to introduce variances into model seem to cause
  problems I now try another slightly modified model introduced by Aalto
  university researchers (Cho et al. 2011).

  Now the energy of the system is defined to be\ 

  <\math>
    E<rsub|G*B>(v,h)

    =<frac|1|2><big|sum><rsub|i>(v<rsub|i>-a<rsub|i>)<rsup|2>/\<sigma\><rsup|2><rsub|i>-<big|sum><rsub|i,j><frac|v<rsub|i>|\<sigma\><rsup|2><rsub|i>>w<rsub|i*j>h<rsub|j>-<big|sum><rsub|i>b<rsub|i>h<rsub|i>

    =<frac|1|2>(v-a)D<rsup|-2>(v-a)-v<rsup|T>D<rsup|-2>W*h-b<rsup|T>h

    =<big|sum><rsub|i><frac|1|2>*v<rsup|2><rsub|i>/\<sigma\><rsup|2><rsub|i>+<frac|1|2>a<rsup|2><rsub|i>/\<sigma\><rsup|2><rsub|i>-<big|sum><rsub|i>(a<rsub|i>+<big|sum><rsub|j>w<rsub|i*j>h<rsub|j>)v<rsub|i>/\<sigma\><rsup|2><rsub|i>-<big|sum><rsub|i>b<rsub|i>h<rsub|i>

    =<frac|1|2><big|sum><rsub|i>(v<rsub|i>-(a<rsub|i>+<big|sum><rsub|j>w<rsub|i*j>h<rsub|j>))<rsup|2>/\<sigma\><rsup|2><rsub|i>+<frac|1|2><big|sum><rsub|i>a<rsup|2><rsub|i>/\<sigma\><rsup|2><rsub|i>-<frac|1|2><big|sum>(a<rsub|i>+<big|sum><rsub|j>w<rsub|i*j>h<rsub|j>)<rsup|2>-<big|sum><rsub|i>b<rsub|i>h<rsub|i>

    =<frac|1|2>(v-(a+W*h))<rsup|T>D<rsup|-2>(v-(a+W*h))+<frac|1|2>a<rsup|T>D<rsup|-2>a-<frac|1|2>\<\|\|\>a+W*h\<\|\|\><rsup|2>-b<rsup|T>h
  </math>

  After this it is again quite straightforward to calculate the conditional
  probabilities given hidden or visible neurons.

  <math|P(v\|h)=<frac|e<rsup|-<frac|1|2><big|sum><rsub|i>(v<rsub|i>-(a<rsub|i>+<big|sum><rsub|j>w<rsub|i*j>h<rsub|j>))<rsup|2>/\<sigma\><rsup|2><rsub|i>><rsup|>|<big|int>e<rsup|-<frac|1|2><big|sum><rsub|i>(v<rsub|i>-(a<rsub|i>+<big|sum><rsub|j>w<rsub|i*j>h<rsub|j>))<rsup|2>/\<sigma\><rsup|2><rsub|i>*>*d*v>\<propto\>><with|mode|math|e<rsup|-<frac|1|2><big|sum><rsub|i>(v<rsub|i>-(a<rsub|i>+<big|sum><rsub|j>w<rsub|i*j>h<rsub|j>))<rsup|2>/\<sigma\><rsup|2><rsub|i>>\<sim\><big|prod><rsub|i>N(><with|mode|math|a<rsub|i>+<big|sum><rsub|j>w<rsub|i*j>h<rsub|j>,\<sigma\><rsup|2><rsub|i>)>

  <strong|><strong|<math|P(v\|h)\<sim\>Normal(a+W*h,D<rsup|2>)>>,
  <strong|<strong|<strong|<math|D<rsup|2>=diag[\<sigma\><rsup|2><rsub|1>\<ldots\>\<sigma\><rsup|2><rsub|D>]>>>>

  <strong|<strong|>><strong|>

  <math|P(h<rsub|j>\|v)=<frac|e<rsup|<big|sum><rsub|i,j><frac|v<rsub|i>|\<sigma\><rsup|2><rsub|i>>w<rsub|i*j>h<rsub|j>+<big|sum><rsub|j>b<rsub|j>h<rsub|j>><rsup|><rsup|>|<big|sum><rsub|h<rsub|j>>e<rsup|<big|sum><rsub|i,j><frac|v<rsub|i>|\<sigma\><rsup|2><rsub|i>>w<rsub|i*j>h<rsub|j>+<big|sum><rsub|j>b<rsub|j>h<rsub|j>>>=<frac|e<rsup|(<big|sum><rsub|i><frac|v<rsub|i>|\<sigma\><rsup|2><rsub|i>>w<rsub|i*j>+b<rsub|j>)h<rsub|j>><rsup|><rsup|>|<big|sum><rsub|h<rsub|j>>e<rsup|(<big|sum><rsub|i><frac|v<rsub|i>|\<sigma\><rsup|2><rsub|i>>w<rsub|i*j>+b<rsub|j>)h<rsub|j>><rsup|>>=sigmoid(><with|mode|math|<big|sum><rsub|i><frac|v<rsub|i>|\<sigma\><rsup|2><rsub|i>>w<rsub|i*j>+b<rsub|j>>)

  <strong|<math|P(h\|v)=sigmoid(b+v<rsup|T>D<rsup|-2>W)>>

  \;

  Notice here that this time the mean is not related to variances which can
  be useful.

  \;

  After calculating the conditional probabilities, we again calculate free
  energy of the energy function (vand then calculate its partial derivates.

  <\with|mode|math>
    F(v)=-log<big|sum><rsub|h>e<rsup|-E<rsub|G*B>(v,h)>=-log<big|sum><rsub|h>e<rsup|-<frac|1|2>(v-a)<rsup|T>D<rsup|-2>(v-a)+v<rsup|T>D<rsup|-2>W*h+b<rsup|T>h>

    =<rsup|><frac|1|2>(v-a)<rsup|T>D<rsup|-2>(v-a)-<big|sum><rsub|i>log<big|sum><rsub|h<rsub|i>>e<rsup|(W<rsup|T>*D<rsup|-2>**v+b)<rsub|i>h<rsub|i>>

    =<frac|1|2>(v-a)<rsup|T>D<rsup|-2>(v-a)-<big|sum><rsub|i>log(1+e<rsup|(W<rsup|T>*D<rsup|-2>**v+b)<rsub|i>>)

    =<frac|1|2><big|sum><rsub|i><frac|(v<rsub|i>-a<rsub|i>)<rsup|2>|\<sigma\><rsup|2><rsub|i>>-<big|sum><rsub|i>log(1+exp(<big|sum><rsub|j>w<rsub|j*i>v<rsub|j>/\<sigma\><rsup|2><rsub|j>+b<rsub|i>))
  </with>

  \;

  From this formulation we can calculate

  <\math>
    <frac|\<partial\>F|\<partial\>a>=-D<rsup|-2>(a-v)
  </math>

  <math|<frac|\<partial\>F|\<partial\>b<rsub|i>>=-<frac|e<rsup|(W<rsup|T>*D<rsup|-2>**v+b)<rsub|i>>|1+e<rsup|(W<rsup|T>*D<rsup|-2>**v+b)<rsub|i>>>=-sigmoid(W<rsup|T>D<rsup|-2>v+b)<rsub|i>>

  <math|<frac|\<partial\>F|\<partial\>b>=-sigmoid(W<rsup|T>D<rsup|-2>v+b)>

  <math|<frac|\<partial\>F|\<partial\>w<rsub|i*j>>=-<frac|e<rsup|(W<rsup|T>*D<rsup|-2>**v+b)<rsub|i>>|1+e<rsup|(W<rsup|T>*D<rsup|-2>**v+b)<rsub|i>>>*(*v<rsub|j>/\<sigma\><rsup|2><rsub|j>)=-<frac|v<rsub|j>|\<sigma\><rsub|j><rsup|2>>*sigmoid(W<rsup|T>D<rsup|-2>v+b)<rsub|i>>

  \;

  But the crucial derivate is the derivate of <math|\<sigma\><rsub|i>> terms,
  we alter energy function by the change of terms:
  <math|1/\<sigma\><rsup|2><rsub|i>=e<rsup|-z<rsub|i>>=\<alpha\>> leading
  into formula:

  <with|mode|math|F(v)=<frac|1|2><big|sum><rsub|i>(v<rsub|i>-a<rsub|i>)<rsup|2>*e<rsup|-z<rsub|i>>-<big|sum><rsub|j>log(1+exp(<big|sum><rsub|i>w<rsub|i*j>v<rsub|i>*e<rsup|-z<rsub|i>>+b<rsub|j>))>

  And then derivate with respect to <math|z<rsub|i>>:

  <\math>
    <frac|\<partial\>F|\<partial\>z<rsub|i>>=-<frac|1|2>(v<rsub|i>-a<rsub|i>)<rsup|2>*e<rsup|-z<rsub|i>>+<big|sum><rsub|j><frac|exp(<big|sum><rsub|i>w<rsub|i*j>v<rsub|i>*e<rsup|-z<rsub|i>>+b<rsub|j>)|1+exp(<big|sum><rsub|i>w<rsub|i*j>v<rsub|i>*e<rsup|-z<rsub|i>>+b<rsub|j>)>*w<rsub|i*j>*v<rsub|i>*e<rsup|-z<rsub|i>>

    =-e<rsup|-z<rsub|i>>*[<frac|1|2>(v<rsub|i>-a<rsub|i>)<rsup|2>*-v<rsub|i><big|sum><rsub|j>w*<rsub|i*j>sigmoid(W<rsup|T>D<rsup|-2>v+b)<rsub|j>]
  </math>

  \;

  This is the exactly same formula as the one given in the paper of Aalto uni
  people (<em|Improved Learning of Gaussian-Bernoulli Restricted Boltzmann
  Machines. ICANN 2011.>)

  \;

  <strong|Implementation of alternative Gaussian model>

  As previously, for prototyping purposes, only Octave code in
  src/neuralnetworks/octave/ was developed initially.

  THERE SEEMS TO BE PROBLEMS WITH THIS IMPLEMENTATION AND MY CIRCLE TEST
  DATA. IT SIMPLY ESTIMATES ONLY MEAN VALUE AND VARIANCE (SINGLE CLUSTER) OF
  THE DATA INSTEAD OF MULTIPLE CLUSTERS TO COVER THE CIRCLE WITH PROPER
  VARIANCE.

  <strong|TODO:>

  - download test pictures (MNIST) and test methods with grayscale picture
  data.. TEST IF THE AALTO PEOPLE METHOD WORKS WITH PIXEL DATA WHERE THERE IS
  TYPICALLY ONLY A SINGLE MEAN THAT MAKES SENSE (and you reproduce their
  paper's results otherwise)

  - download some other grayscale pictures and test if the methods work
</body>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
  </collection>
</references>