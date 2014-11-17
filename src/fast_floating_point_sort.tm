<TeXmacs|1.0.1.21>

<style|generic>

<\body>
  \;

  <strong|Existence of <with|mode|math|O(N)> Sorting Algorithm for Any Finite
  Sets>

  Tomas Ukkonen

  \;

  This is mapping of the problem into set of integers (and back) in
  <with|mode|math|O(n)> time. This is especially useful when dealing with
  floating point numbers (after one have figured out efficient mapping).

  \;

  <strong|1. Theoretical proof>

  Proof of algorithm requires a lemma given below which can be understood to
  be true by pure intuitively means but in order to be mathematically sound
  it has been proofed from certain simple basic assumptions. Idea of (very
  simple) algorithm itself has been given after the proof of lemma. In fact
  it's so simple that it's amazing how (at least) common textbooks don't talk
  about (if this is publicly known at all).

  \;

  <strong|Lemma>

  Let <with|mode|math|F<rsub|1>> and <with|mode|math|F<rsub|2>> to be two
  finite sets of numbers of same size with 'nice' ordering properties (see
  below). Then there exists isomorphism <with|mode|math|f:F<rsub|1>\<rightarrow\>F<rsub|2>>
  with respect to <with|mode|math|\<less\>> operator such that if
  <with|mode|math|x<rsub|1>\<less\>x<rsub|2>\<Leftrightarrow\>f(x<rsub|1>)\<less\>f(x<rsub|2>)>,
  <with|mode|math|x<rsub|!,>x<rsub|2>\<in\>F<rsub|1>> and
  <with|mode|math|f(x<rsub|1>)>, <with|mode|math|f(x<rsub|2>)>
  <with|mode|math|\<in\>F<rsub|2>>.

  \;

  <with|mode|math|N>-bit long floating point numbers form a finite set of
  numbers <with|mode|math|F>, <with|mode|math|\|F\| = 2<rsup|N>>. This
  <with|mode|math|F> has following ordering properties (equivalent to well
  ordering?):\ 

  there exists minimum (first) element:\ 

  (1) <with|mode|math|\<exists\> m<rsub|F>\<in\>F:>,
  <with|mode|math|m<rsub|F>\<less\>m, \<forall\> m\<in\>F,m\<neq\>m<rsub|F>>.

  (2) and every non-empty subset of <with|mode|math|F> have this property
  too.

  (3) either <with|mode|math|n\<less\>m> or <with|mode|math|m\<less\>n> for
  <with|mode|math|\<forall\> m,n\<in\>F>, <with|mode|math|m\<neq\>n>. And if
  <with|mode|math|a\<less\>n> and <with|mode|math|n\<less\>m> then
  <with|mode|math|a\<less\>m>. <with|mode|math|a\<in\>F>.

  [ TODO/NOTE: (1) and (2) are redundant. (1) follows from finiteness and (3)
  and (2) follows when

  same result is applied to subset of <with|mode|math|F> (which is then also
  finite). This means (with a trivial proof that finite set of integers has
  property (3)) the result of this paper hold for any finite set with
  meaningful <with|mode|math|\<less\>> comparision operator. (as defined by
  (3)) ]

  \;

  From (2) it follows that each <with|mode|math|m\<in\>F> has unique smallest
  follower, unless <with|mode|math|m> is largest element of
  <with|mode|math|F>.\ 

  Proof.

  let's define <with|mode|math|F(m)={n\|n\<gtr\>m \<vee\> n\<in\>F}> and
  assume (for now) that it's non-empty.

  Now set <with|mode|math|G(m)={n-m\| n\<in\>F(m)}> has smallest element
  because of (2).

  and finally let's define biggest value of
  <with|mode|math|><with|mode|math|M<rsub|F>\<in\>F> to be number with
  property \ <with|mode|math|F(M<rsub|F>)={}>.

  \;

  This number if unique: if there exists <with|mode|math|M<rsup|<rprime|'>><rsub|F>>
  with <with|mode|math|F(M<rsub|F><rsup|<rprime|'>>)={}> , then by property
  (3). Either <with|mode|math|M<rsub|F><rsup|<rprime|'>>\<gtr\>M<rsup|><rsub|F>>
  or <with|mode|math|M<rsub|F>\<gtr\>M<rsup|<rprime|'>><rsub|F>> , but this
  means that either <with|mode|math|F(M<rsub|F><rsup|<rprime|'>>)\<neq\>{}\<vee\>F(M<rsub|F>)\<neq\>{}>
  - a contradiction.

  \;

  Existence of <with|mode|math|M<rsub|F>>: because \ for each number
  <with|mode|math|m<rsub|n>> there exists unique follower
  <with|mode|math|m<rsub|n+1>>:\ 

  <with|mode|math|\|F(m<rsub|n>)\|-1=\|F(m<rsub|n+1>)\|>. Now
  <with|mode|math|F(m<rsub|F>)=2<rsup|N>-1>. Now let's form a sequence of
  sizes of\ 

  (<with|mode|math|F(m<rsub|F>),\<ldots\>,F(m<rsub|n>),F(m<rsub|n+1>),\<ldots\>.>).
  Initial value of sequence is finite and sequence is strictly decreasing,
  from this it follows that for some <with|mode|math|m> condition
  <with|mode|math|F(m)=0> is true.

  \;

  Now lets define bijection mapping <with|mode|math|f:F<rsub|1>\<rightarrow\>F<rsub|2>>
  between two finite sets sharing properties described above. Additionally
  <with|mode|math|\|F<rsub|1>\|=\|F<rsub|2>\|>.

  We construct <with|mode|math|f> iteratively for subsets of
  <with|mode|math|F<rsub|1>> and <with|mode|math|F<rsub|2>>.

  <em|Initial step>

  Set <with|mode|math|B<rsup|1><rsub|1>={m<rsub|F<rsub|1>>}> and
  <with|mode|math|B<rsub|2><rsup|<rsup|>1>={m<rsub|F<rsub|2>>}> and
  <with|mode|math|B>:s share the 'minimum-set' property that they contain all
  smaller numbers from <with|mode|math|F> and that numbers which are in
  respective <with|mode|math|B> set.

  Now <with|mode|math|B<rsup|1><rsub|1>> and
  <with|mode|math|B<rsup|2><rsub|2>> have trivial isomorphism
  <with|mode|math|f<rsup|1>(m<rsub|F<rsub|1>>)=m<rsub|F<rsub|2>>>. And they
  have also have 'minimum-set' property.

  <em|Inductive step>

  Assume isomorphism <with|mode|math|f<rsup|n>> between
  <with|mode|math|B<rsup|n><rsub|1>\<in\>F<rsub|1>> and
  <with|mode|math|B<rsup|n><rsub|2>\<in\>F<rsub|2>> exists and that largest
  element (which exists and is unique as proofed for subsets of
  <with|mode|math|F>). Now if this either have smallest following or it
  doesn't have. This means that largest elements of
  <with|mode|math|F<rsub|x>> belongs in <with|mode|math|B<rsub|x>> , but
  because of 'minumum-set' property it means
  <with|mode|math|B<rsub|x>=F<rsub|x>>. Additionally because
  <with|mode|math|\|B<rsub|1>\|=\|B<rsub|2>\|=\|F<rsub|x>\|> and because
  <with|mode|math|F<rsub|1>> and <with|mode|math|F<rsub|2>> have same size:
  <with|mode|math|\|B<rsub|1>\|=\|F<rsub|1>\|> and
  <with|mode|math|\|B<rsub|2>\|=\|F<rsub|2>\|>. This means isomorphism for
  the whole set exists.

  Now assume that there exists smallest follower <with|mode|math|m<rsub|x>>
  for both <with|mode|math|B<rsub|x>>.

  By forming new sets <with|mode|math|B<rsup|n+1><rsub|x>=B<rsup|n><rsub|x>\<cup\>{m<rsub|x>}>.
  And defining extension to isomorphism <with|mode|math|f<rsub|><rsub|><rsup|n+1>(m<rsub|1>)=m<rsub|2>>,
  <with|mode|math|f<rsub|<rsup|>><rsup|n+1>(m)=f<rsup|n>(m),
  m\<neq\>m<rsub|x>>. <with|mode|math|f<rsup|n+1>> is clearly still
  bijection. And additionally <with|mode|math|B<rsub|x><rsup|n+1>> have still
  'minimum-set' property because all values of
  <with|mode|math|B<rsup|n><rsub|x>> are smaller than
  <with|mode|math|m<rsub|x>>.\ 

  Proof of isomorphism property:

  if <with|mode|math|m,n\<in\>F<rsub|1><rsup|n>>, then clearly isomorphism
  property exists. Now if <with|mode|math|m=m<rsub|1>> just added new value
  then:

  <with|mode|math|n\<less\>m<rsub|1>> and because
  <with|mode|math|f<rsub|><rsup|n+1>(n)\<in\>B<rsub|2><rsup|n>> it follows
  <with|mode|math|f<rsup|n+1>(n)\<less\>m<rsub|2>=f<rsup|n+1>(m<rsub|1>)>.
  This proofs function is morphism.

  Clearly in above argumentation <with|mode|math|f<rsup|n+1>> can swapped
  with it inverse and sets <with|mode|math|F<rsub|1>> and
  <with|mode|math|F<rsub|2>> can be also swapped. So
  <with|mode|math|f<rsup|n+1>> is isomorphism.

  This constructs isomorphism explicitely and therefore proofs lemma.

  \;

  <strong|Algorithm>: <strong|General radix sort>

  Now that detailed proof of given simple lemma has been given proof is
  simple. Assume <with|mode|math|F<rsub|1>> is set of floating point numbers
  and <with|mode|math|F<rsub|2>> is set of integers. Calculating isomorpism
  <with|mode|math|f> or it's inverse takes time <with|mode|math|O(1)> for
  single value and <with|mode|math|O(N)> for the whole set of numbers. Radix
  sort with integers can be performed in <with|mode|math|O(N)>. So total time
  for mapping numbers, radix sorting and inverse mapping back to floating
  point numbers is <with|mode|math|O(N)>.

  There are certain small technical detail with must be taken care for
  practical implementation: AFAIK ordering of <with|mode|math|NaN> with
  respect to other numbers in IEEE floating point standard haven't been
  defined. This can be remedied by explicitely define <with|mode|math|NaN> to
  be largest floating point number that is even bigger than +Inf. After this
  definition IEEE floating point numbers form a set with 'nice' ordering
  properties as described in a proof of lemma. Clearly those properties are
  satisfied also any set of by <with|mode|math|n->digit (unsigned) integers.

  \;

  <strong|2. Practical Implementation>

  \;

  Calculating mapping between floating point numbers and unsigned integers
  take certain amount of time with cannot been seen from assumptotic
  big-<with|mode|math|O> notation.\ 

  Implementation in <em|converion.cpp>, <em|conversion.h>,
  <em|tst/conv_test.cpp> shows that radix sort for floating point numbers
  with in-between conversion is faster than quicksort in AMD Athlon 500 MHz
  when there are aprox 500 000 floating point numbers. After this one radix
  floating point sort will quickly become much faster than quicksort due to
  better assymptotic behaviour: <with|mode|math|O(n)> vs.
  <with|mode|math|O(n*log*n)>.

  \;

  <strong|References>

  IEEE 754: Stard for Biary Floating Point Arithmetic

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;
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
