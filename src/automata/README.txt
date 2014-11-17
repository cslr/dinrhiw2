
 finite state automata, non-deterministic

PLAN - DO NON-DETERMINISTIC AUTOMATA FIRST
THEN ADD REGEXP MATCHING


CLASSES

namespace automata

Automata
contains list of states and edges
regexp string has form:
"[(abc)+]( )*=( )*[.+];( )*%.*" accepts "abcabc =   niksu ; % comment",
marked substrings are 
mark0 = (abc)+
mark1 = .+

"([.+]( )*=( )*[.+];( )*)*%.*" accepts "house = "AA"; A = [1 2; 3 4]; % blah
mark00 = house
mark01 = "AA"
mark10 = A
mark11 = [1 2; 3 4]

 - regexp matching is greedy

State
state has list of leaving edges and
possible actions. (for parsing etc.)

Edge
edges contain condition about next character
