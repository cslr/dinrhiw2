<TeXmacs|1.0.7.18>

<style|generic>

<\body>
  <strong|Some notes about my RBM implementation (binary RBM)>\ 

  Tomas Ukkonen, 2015

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

  <with|font-series|bold|Continuous RBM>

  TODO: implement <with|font-series|bold|continuous RBM> similarly to what is
  described in the <verbatim|CRBM_iee2003.pdf> paper.
</body>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
  </collection>
</references>