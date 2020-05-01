/*
 * interface for time estimation classes
 *
 */

#ifndef ETA_h
#define ETA_h

namespace whiteice
{
  
  // eta estimators should be thread safe
  // ( estimate() and either start() or update()
  //   may be called at same time by more than one thread) 
  template <typename T>
    class ETA
    {
    public:
      virtual ~ETA(){ }
      
      virtual bool start(T begin, T end)  = 0;
      virtual bool update(T current)  = 0;
      
      // ETA in seconds when the end value will be reached
      virtual T estimate() const  = 0;
      
    };
  
};


#endif
