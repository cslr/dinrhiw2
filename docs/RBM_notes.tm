<TeXmacs|1.0.7.18>

<style|generic>

<\body>
  <strong|Some notes about my RBM implementation>

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
  care about the weight matrix <math|\<b-W\>> and can ignore other terms.
  However, this approach has one theoretical/practical problem. In practice,
  the final hidden state should be always equal to <math|1> but in practice
  it is not because of our calculations set another values to the last hidden
  state.

  Because of this, another, alternative approach was/will be tried. Only
  visible terms are extended to have <math|1> and the hidden layer DOES NOT
  have extra variables. This should be also compatible with the stacking of
  RBMs.

  Stacking of RBMs:

  <\itemize-dot>
    <item>the next layer (hidden layer) is not extended to have extra
    variables but the input layer (visible) is always extended to have extra
    <math|1> meaning that the last weight matrix terms are the bias-terms
    like in ``normal'' neural network

    <item>after learning a single layer RBM, hidden layer variables are again
    calculated and <em|again extended> with ones. This means that there is
    NEVER backward bias-terms but only forward bias-terms like in normal
    feedforward neural networks

    <item>after calculating <math|k> layers of RBMs using a special function,
    they are combined to form a <strong|<em|deep belief network> - DBN>.
  </itemize-dot>

  Calculating deep neural network (TODO):

  <\itemize-dot>
    <item>write code to calculate <with|font-series|bold|DBN> using
    discretized 0/1-input data and restricted boltzmann machines, if possible
    extend RBM to work with continuous data if possible?

    <item>write code that generates neural network with a logistic
    non-linearity similarly to RBMs and uses bias terms introduced into RBMs
    through artificial all <math|1> inputs in the visible nodes.
  </itemize-dot>

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

  <center|<math|\<b-W\><rsup|T>=><center|<block|<tformat|<table|<row|<cell|+5.3>|<cell|+0.1>>|<row|<cell|+5.3>|<cell|+0.1>>|<row|<cell|-0.3>|<cell|-2.5>>|<row|<cell|-0.3>|<cell|-2.5>>|<row|<cell|-9.0/0.0>|<cell|+2.6>>>>>>>

  From this we can see that the first two terms are exactly same as the first
  term of the hidden layer and that the 3rd and 4th term are inverse of the
  second term in hidden layer. So if the hidden layer term is <math|1> then
  the visible layer term is zero (large negative value) but if the term is
  zero, then it is maybe ALSO zero because the value of the sigmoid function
  is close zero and the probability of selecting either state is close to 50%
  (with some negative correlation to the statistically independent first
  term!). Finally, if the bias term weights seem to chosen randomly for the
  first term, the value is negative of the first term or positive of the
  second term but is not certainly close to <math|1> which is should always
  be thereby generating spurious results.

  It is also possible to look at the weight matrix <math|\<b-W\>> that
  generates hidden RBM states:

  <center|<math|\<b-W\>=><block|<tformat|<table|<row|<cell|-5.5>|<cell|-5.5>|<cell|0.7>|<cell|0.7>|<cell|-0.2>>|<row|<cell|-0.3>|<cell|-0.3>|<cell|-2.4>|<cell|-2.4>|<cell|2.57>>>>>>

  Here we can see that the first hidden state is <math|1> if the first state
  <math|X> is zero and the second term is <math|X> is
  <math|<wide|X|\<bar\>>>. Again, the second state is <math|1>, if <math|Y=0>
  and <math|0> is <math|Y=1>.

  <em|These results do not look that good..> Thereby another, slightly
  altered algorithm was again used that DO NOT set extended values of
  <math|X> always to <math|1> during stimulation step. THIS LEAD TO EVEN
  WORSE RESULTS. The algorithm just CANNOT FIND 2 CLEAR HIDDEN STATES IT
  SHOULD.

  SO INSTEAD, I THEN TRIED TO INCREASE NUMBER OF HIDDEN STATES TO 8 STATES.

  \;

  TODO: implement <with|font-series|bold|continuous RBM> as described in the
  <verbatim|CRBM_iee2003.pdf> paper.
</body>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
  </collection>
</references>