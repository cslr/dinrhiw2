

#include "Factory.h"
#include "FactoryImpl.h"
#include "vertex.h"

using namespace whiteice;


int main(int argc, char **argv)
{
  Factory< vertex<double> > *F;  
  // here interface and impl. is same
  F = FactoryImpl< vertex<double>, vertex<double> >::getInstance();
  
  /* after setup generic factory can be used
   * to create instances of orig. class interfaces
   * without knowledge about the actual implementation
   * interface. - it's not possible to create implementation
   * in any other way.
   * - factory can also easily used to force/create memory
   *   collection/freeing etc. (as in java) - but not really
   *   wanted (should be using java)
   * - factory interface can be easily passed to other functions
   *   as way of creating interface implementations without
   *   anyway to know/access/use the specific impl.  
   *   passing factories generating interface instances leads
   *   to very generic design and actual implementation
   *   can be easily changed without (almost)
   *   no changes in elsewhere but in implementation.
   *   (for example: generic matrix, hardware platform optimized matrix
   *    and FactoryImpl is chosen when program starts and everything
   *    else uses Factory<matrix(Interface)> )
   */

  vertex<double> *i = F->createInstance();
  
  delete i;

#if 0

  // some notes/ TODO:
  //
  // in order FM::set() to be safe 
  // accesses via Singleton should be restricted to classes
  // with 'key'.
  // -> implement PrivateSingleton, no global access to singleton class
  // (what is normally thought to be singleton class)
  // SecureSingleton which inherits SecureServer class (aspect)
  // and user must be SecureClient class  and accesses must be
  // be made via SecureClient's private call which uses Authentication
  // interface class (usually very simple random number matching
  // in internal code and cryptographically secure in network code
  // (for example when CORBA based name and directory services code (CNDI)
  //  works add own authentication code to it (or did CORBA have some kind of
  //  internal secure authentication suport?)
  // 

  interface *v;
  v = FM::create(v); // not so bad when compared to

  interface *v = new Interface();
  
  // make syntax easier, now:
  v= FactoryMapping::getInstance()->createInstance(v);
  // for example
  v = FM::create(v); // interface for factory mapping
  // where interface is implemented with registered FactoryImpl
  // and direct creation don't work - works everywhere, setup
  // implementation for interface once - maximizes coding for interface.
  // FactoryImpls are free'ed automatically as well as FactoryMapping (also allocation)
  delete v;
#endif
  
  return 0;  
}

