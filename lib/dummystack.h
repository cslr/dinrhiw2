

#ifndef dummystack_h
#define dummystack_h

#include "stack.h"

namespace whiteice
{

  template <typename D, typename T=int>
    class dummystack : public stack<D,T>
  {
    public:
    
    dummystack(){ size_ = 0; }
    
    bool push(const D& d){ printf("push\n"); return true; }
    D pop(){ printf("bepop\n"); return 0; }
    
    const T& size() const{ return size_; }
    
    private:
    
    T size_;
    
  };
};

#endif

