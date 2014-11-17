/*
 * function class interface
 * useful when passing different kind of functions to
 * other classes
 */

#ifndef function_h
#define function_h

#ifdef __GNUG__
#ifdef PURE_FUNCTION
#undef PURE_FUNCTION
#endif

#define PURE_FUNCTION __attribute__ ((pure))

#else

#define PURE_FUNCTION

#endif


namespace whiteice
{

  template <typename T, typename U>
    class function
    {
    public:
      
      virtual ~function(){ }
      
      // calculates value of function
      virtual U operator() (const T& x) const PURE_FUNCTION = 0;
      
      // calculates value
      virtual U calculate(const T& x) const PURE_FUNCTION = 0;
      
      // calculates value 
      // (optimized version, this is faster because output value isn't copied)
      virtual void calculate(const T& x, U& y) const = 0;
      
      // creates copy of object
      virtual function<T,U>* clone() const = 0;
    };
  
}



#endif




