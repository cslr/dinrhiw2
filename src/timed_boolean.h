/*
 * simple timed boolean variables: (look libnazgul for better ones)
 * implementation for boolean variables,
 * which will be inverted after given time period
 *
 * measured time is real world time.
 */


#ifndef timed_boolean_h
#define timed_boolean_h


namespace whiteice
{
  
  class timed_boolean
  {
  public:
    
    /* time is time in seconds and initial is
     * value of the variable
     * (if time <= 0, invertion won't be ever done)
     */
    timed_boolean(double time, bool initial = true);
    timed_boolean(const timed_boolean& tb);
    ~timed_boolean();
    
    
    timed_boolean& operator=(const bool bValue);
    timed_boolean& operator=(const timed_boolean& tb);
    
    // compares timed_boolean() against boolean value
    bool operator==(const bool bValue) const ;
    bool operator!=(const bool bValue) const ;

    bool operator==(const timed_boolean& b) const ;
    bool operator!=(const timed_boolean& b) const ;
    
    bool operator!() const ; // returns inverted value
    
    /* returns time left for the change of value,
     * zero if timer has been expired and
     * negative value if calculating remaining time failed.
     */
    double time_left() const ;
    
  protected:
    
    void update() const ;
    
    bool get_time(double& t) const ;
    
  private:
    mutable bool inverted;
    mutable bool variable;
    
    double t1; // invertation time
  };
  
  
  std::ostream& operator<<(std::ostream& ios,
			   const whiteice::timed_boolean&);
  
};



#endif

