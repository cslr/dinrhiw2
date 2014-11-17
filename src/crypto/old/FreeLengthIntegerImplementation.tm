<TeXmacs|1.0.1.16>

<style|generic>

<\body>
  <strong|Free length Positive Integer Code Implementation>

  Brief Documentation and Design of Free Length Positive Integers
  Implementation

  Tomas Ukkonen

  \;

  Integer implementation in <em|nbits_integer.h> and <em|nbits_integer.cpp>
  is based on radix-base representation of\ 

  integer. This means that each integer <with|mode|math|i> value has been
  represented in some <with|mode|math|b> - radix form:

  <\with|paragraph mode|center>
    <with|mode|math|i=<big|sum><rsup|N><rsub|i=0>a<rsub|i>*b<rsup|i>>
  </with>

  And number and degree of basis exponents <with|mode|math|{b<rsup|i>}> are
  decreased and/or increased when it's necessary\ 

  (result is too big or slow). The biggest degree of basis function is scale
  of the function <with|mode|math|scale(i)=N>.

  \;

  <strong|Addition>

  Addition can be done in <with|mode|math|O(max(scale(a),scale(b)))=O(N)>
  time by trivial summing of basises and overflowing too large values to
  higher degree basis whenever necessary.

  <strong|Subtraction>

  Subtraction isn't considered here. Making negative number available from
  free length positive integers implementation isn't that hard.

  <strong|Multiplication>

  Notice that\ 

  <with|mode|math|r*s=<big|sum><rsup|degree(r)+degree(s)><rsub|i=0>r<rsub|i>*s<rsub|(N-i)>*b<rsup|i>>.

  Where coefficients are extended to be zero with negative indexes and with
  values larger than original degree of number. This is modular convolution
  with enlarged coefficients. By using discrete fourier transform
  multiplication can be then implemented with <with|mode|math|O(N*log N)>
  time. However, there maybe problem with precision of floating point number
  used in the Fourier tranform + I don't have free length FFT implemented
  (yet: free-radix-FFT and prime-len-FFT are not done). These matters must be
  solved in order to have efficient implementation. Size of the basis should
  probably be size of the largest integer which machine can use (word). With
  Intel processors either assembly 32bit addition + carry check or 'use only
  31bit and use comparision for detecting overflow into 32th bit in C'\ 

  Slower direct <with|mode|math|O(N<rsup|2>)> algorithm is used currently.

  \;

  <strong|Division>

  Division will be done in self-invented (maybe reinvented) Iterative Linear
  Improvement (see Appendix) algorithm which should converge into correct
  value in <with|mode|math|O(log<rsub|2> b<rsup|N>)=O(N)=O(log<rsub|b> n)>
  time. Where <with|mode|math|N> is degree of the number <with|mode|math|n>.

  \;

  <strong|Modulo>

  Modulo are calculated by using division: <with|mode|math|O(log<rsub|b> n)>.

  \;

  \;

  <strong|Appendix>

  <strong|Iterative Linear Improvement Algorithm>

  Let problem be to solve <with|mode|math|<frac|a|b>=c> for integers
  <with|mode|math|a,b,c\<in\>\<bbb-N\>>.\ 

  Clearly when <with|mode|math|1\<leqslant\>b\<leqslant\>a> result
  <with|mode|math|c \<in\> [0,a]=[l<rsub|0>,h<rsub|0>]>.

  Lets start by guessing the correct value for the division to be
  <with|mode|math|c<rsub|0>>,\ 

  a good guess might be to choose \ <with|mode|math|c<rsub|0>=<frac|h<rsub|0>-l<rsub|0>|2>>.

  Then error in made in the division is\ 

  <with|mode|math|d<rsub|0>=a-c<rsub|0>*b>.

  Then the improved guess will be

  <with|mode|math|c<rsub|1>=<frac|d<rsub|0>|b>+c<rsub|0>>, \ which would
  result into exactly right solution.

  Problem is that this requires division which we are trying to solve.

  If we only try to reduct error a little bit then a crude approximation of
  <with|mode|math|<frac|d<rsub|0>|b>> is enough. This will then move solution
  closer to the right direction and approximative correction can be done
  again. This especially useful when we can do division with the limited B
  bits long hardware and use result for the approximation.

  <em|Approximate integer division algorithm>

  Approximation from below

  1. Division by two

  <with|mode|math|<frac|a|2>> can be approximated from the below by dividing
  each coefficient by two separatedly.

  2. Division by any number

  Increase coefficient of <with|mode|math|b>:s highest radix coefficient by
  one, then divide each coefficient of <with|mode|math|a> by this coefficient
  and shift <with|mode|math|a>:s basises by highest basis order of
  <with|mode|math|b>.\ 

  Maximum error: <with|mode|math|a<rsub|k>*<left|[><frac|1|b<rsub|l>+1>*-<frac|1|b<rsub|l>><left|]>B<rsup|k-l>>,
  where <with|mode|math|B> is basis radix.

  \;

  Approximation from above

  1. Division by two

  Add each element +1 and then divide each coefficient by two

  2. Division by any number <with|mode|math|<frac|a|b>>\ 

  <with|mode|math|<frac|a|b>> can be approximated from the above by dividing
  each coefficient of <with|mode|math|a> by highest coefficient of
  <with|mode|math|b> and additionally shifting radixes basises of
  <with|mode|math|a> by highest radix basis of <with|mode|math|b>.

  Maximum error: <with|mode|math|>

  \;

  Convergence proof of the algorithm

  Improvement to the guess can be of the form

  <with|mode|math|c<rsub|n+1>=<mid|lfloor><frac|d<rsub|n>|b><left|rfloor>+c<rsub|n>>

  <with|mode|math|c<rsub|n+1>=<left|lceil><frac|d<rsub|n>|b><left|rceil>+c<rsub|n>>

  \;

  After correction the error will be (<with|mode|math|c<rsub|x>> is the
  correct value)

  Now errors after two successive approximations are:

  <\with|mode|math>
    e<rsub|n+1>

    =\|c<rsub|n+1>-c<rsub|x>\|

    =\|aprox(<frac|d<rsub|n>|b>)+c<rsub|n>-[<frac|d<rsub|n>|b>+c<rsub|n>]\|

    =\|aprox(<frac|d<rsub|n>|b>)-<frac|d<rsub|n>|b>\|
  </with>

  Now error when approximating from the below is order of
  <with|mode|math|B<rsup|degree(d<rsub|n>)-degree(b)>=degree(d<rsub|n+1>)>
  this means degree of <with|mode|math|d*<rsub|n>> becomes finally smaller
  than <with|mode|math|degree(b)> which means
  <with|mode|math|<frac|d<rsub|x>|b>=0> and there's no error. Therefore
  algorithm which uses approximation from the below will always converge.

  \;

  \;

  ======================

  <\with|mode|math>
    e<rsub|n+2>

    =\|c<rsub|n+2>-c<rsub|x>\|

    =\|aprox(<frac|d<rsub|n+1>|b>)+c<rsub|n+1>-c<rsub|x>\|

    =\|aprox(<frac|d<rsub|n+1>|b>)+aprox(<frac|d<rsub|n>|b>)-<frac|d<rsub|n>|b>\|\<leqslant\>\|aprox(<frac|d<rsub|n+1>|b>)\|+\|aprox(<frac|d<rsub|n>|b>)-<frac|d<rsub|n>|b>\|
  </with>

  <with|mode|math|\|e<rsub|n+2>-e<rsub|n+1>\|\<leqslant\>
  \|aprox(<frac|d<rsub|n+1>|b>)\|>.

  \;

  <\with|mode|math>
    e<rsub|\<infty\>>=\|<big|sum><rsub|n>aprox(<frac|d<rsub|n>|b>)-<frac|d<rsub|0>|b>
    \|
  </with>

  ===============
</body>

<\initial>
  <\collection>
    <associate|paragraph width|6.5in>
    <associate|odd page margin|1in>
    <associate|page right margin|1in>
    <associate|page top margin|1in>
    <associate|reduction page right margin|0.7in>
    <associate|page type|letter>
    <associate|reduction page bottom margin|0.3in>
    <associate|even page margin|1in>
    <associate|reduction page left margin|0.7in>
    <associate|page bottom margin|1in>
    <associate|reduction page top margin|0.3in>
    <associate|language|finnish>
  </collection>
</initial>
