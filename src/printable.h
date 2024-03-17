/*
 * printable interface
 * Tomas Ukkonen <tomas.ukkonen@hut.fi>
 *
 *   - for printable objects
 *   - this will be extended later
 *     (code to transform to text and back! succesfully)
 */

#ifndef printable_h
#define printable_h

namespace whiteice
{
  class printable
  {
  public:
    virtual ~printable(){ }
    
    /* prints informative & short values/information
     * about object to stdio/cout */
    virtual bool print() = 0;
  };
  
}


#endif

