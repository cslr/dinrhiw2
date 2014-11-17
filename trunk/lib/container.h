/*
 * container interface
 * Tomas Ukkonen <tomas.ukkonen@iki.fi>
 */

#ifndef container_h
#define container_h

namespace whiteice
{
  
  template <typename D, typename T>
    class container
    {
    public:
      
      virtual ~container(){ }
      
      /* clear()s container from any data */
      virtual void clear() throw() = 0;
      
      /* returns number of elements in a container */
      virtual unsigned int size() const throw() = 0;
      
    };
  
}


#endif

