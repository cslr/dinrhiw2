

#include <iostream>
#include <stdexcept>
#include <exception>
#include "Edge.h"
#include "State.h"

using namespace std;


int main()
{
  bool ok = true;

  try{
    automata::State<char>* s[2];
    automata::Edge<char>* e;
    
    s[0] = new automata::State<char>();
    s[1] = new automata::State<char>();
    e = new automata::Edge<char>('a', *s[1]);
    
    
    if(s[0]->add(*e) == false){
      cout << "adding edge to State s0 failed - FAILURE." << endl;
      ok = false;
    }
    
    ////////////////////////////////////////////////////
    // SIMPLE EDGE TESTS
    
    if(e->accept('a') == false){
      cout << "Edge('a') does not accept 'a' - FAILURE." << endl;
      ok = false;
    }
    

    if(e->accept('b')){
      cout << "Edge('a') accepts 'b' - FAILURE!" << endl;
      ok = false;
    }
    
    
    if(e->constraintlessEdge()){
      cout << "Edge('a') is contraintless - FAILURE!" << endl;
      ok = false;
    }
    
    
    if('a' != e->getConstraint()){
      cout << "Edge('a') returns wrong constraint - FAILURE." << endl;
      ok = false;
    }

  
    const automata::State<char> *r = &(e->getTransitionState());
    
    if(r != s[1]){
      cout << "Edge<char>('a') WRONG TRANSITION STATE - FAILURE." << endl;
      ok = false;
    }

    
    ////////////////////////////////////////////////////
    // SIMPLE STATE CHECKS
    
    if(s[0]->getNumberOfEdges() != 1){
      cout << "s0 state does not have 1 edge - FAILURE." << endl;
      ok = false;
    }
    
    if(s[1]->getNumberOfEdges() != 0){
      cout << "s1 state does have edges - FAILURE." << endl;
      ok = false;
    }
    
    if(s[0]->getEdge(0) != *e){
      cout << "s0 has incorrect edge - FAILURE." << endl;
      ok = false;
    }
        
    delete e;

    ////////////////////////////////////////////////////
    // Edge constraintless edge

    e = new automata::Edge<char>(*(s[0]));
    
    if(e->constraintlessEdge() == false){
      cout << "constraintless edge reports to have constrained edge - FAILURE." << endl;
      ok = false;
    }

    if(e->accept('i') == false){
      cout << "constraintless edge does not accept 'i' - FAILURE." << endl;
      ok = false;
    }
    
    try{
      e->getConstraint();
      cout << "getting constraint for constraintless edge succeeds - FAILURE." << endl;
      ok = false;
    }
    catch(std::logic_error& e){ 
    }
    catch(std::exception& e){
      cout << "getting constraint for constraintless edge throws wrong kind of exception - FAILURE." << endl;
      ok = false;
    }
    
    
    delete e;
    delete s[0];
    delete s[1];
    
  }
  catch(std::exception& e){
    cout << "FAILURE: Unexpected exception: " << e.what() << endl;
    ok = false;
  }

  try{
    // AUTOMATE TEST
    // constructing finite state automate manually for testing
    // "(ab)*[(c|w)+].*=a;"
    // with "ababcc c=a;", "ababcwwa==a;" (ok) and with "abab =a;" and "ababcc c=a; " (failure)
    //

    automata::State<char>* A, *B, *C, *D, *E, *F, *G;
    automata::Edge<char> *ab, *ba, *ad, *ac, *cc, *dd, *cd, *dc, *ce, *de, *ef, *fg;
    automata::Edge<char> *cc2, *dd2;
    
    A = new automata::State<char>();
    B = new automata::State<char>();
    C = new automata::State<char>();
    D = new automata::State<char>();
    E = new automata::State<char>();
    F = new automata::State<char>();
    G = new automata::State<char>();
    
    ab = new automata::Edge<char>('a', *B);
    ba = new automata::Edge<char>('b', *A);
    
    ad = new automata::Edge<char>('w', *D);    
    ac = new automata::Edge<char>('c', *C);
    cc = new automata::Edge<char>('c', *C);
    dd = new automata::Edge<char>('w', *D);

    cc2 = new automata::Edge<char>("=wc", false, *C); // everything - except.
    dd2 = new automata::Edge<char>("=wc", false, *D);

    cd = new automata::Edge<char>('w', *D);
    dc = new automata::Edge<char>('c', *C);

    ce = new automata::Edge<char>('=', *E);
    de = new automata::Edge<char>('=', *E);

    ef = new automata::Edge<char>('a', *F);
    fg = new automata::Edge<char>(';', *G);

    A->add(*ab);
    A->add(*ad);
    A->add(*ac);
    B->add(*ba);

    C->add(*cc);
    C->add(*cc2);
    C->add(*cd);
    C->add(*ce);
    
    D->add(*dd);
    D->add(*dd2);
    D->add(*dc);
    D->add(*de);

    E->add(*ef);
    F->add(*fg);
    
    automata::Automata<char>* a =
      new automata::Automata<char>();
    
    a->add(*A, true);
    a->add(*B);
    a->add(*C);
    a->add(*D);
    a->add(*E);
    a->add(*F);
    a->add(*G, false, true);

    delete A;
    delete B;
    delete C;
    delete D;
    delete E;
    delete F;
    delete G;

    delete ab;
    delete ba;
    delete ad;
    delete ac;
    delete cc;
    delete dd;
    delete cd;
    delete dc;
    delete ce;
    delete de;
    delete ef;
    delete fg;
        
    // automata constructed
    // - testing automata

    // with "ababcc c=a;", "ababcwwa==a;" (ok) and with "abab =a;" and "ababcc c=a; " (failure)
    std::string teststr[4];
    teststr[0] = "ababcc c=a;";
    teststr[1] = "ababcwwa==a;";
    teststr[2] = "abab =a;";
    teststr[3] = "ababcc c=a; ";
    
    
    for(unsigned int i=0;i<4;i++)
    {
      bool result = a->check(teststr[i]);
			     
      cout << "CHECK('" <<  teststr[i] << "') = "
	   << result << endl;
    }
    
    
    delete a;
  }
  catch(std::exception& e){
    cout << "FAILURE: Unexpected exception: " << e.what() << endl;
    ok = false;
  }
  
  

  if(ok)
    cout << "AUTOMATA TESTS PASSED." << endl;
  
  return 0;
}





