/*
 * Additional interface for visualizing scatter plots or other graphics
 * user of library must implement the interface
 *
 */

#ifndef __VisualizationInterface_h
#define __VisualizationInterface_h

#include <vector>
#include "dinrhiw_blas.h"
#include "vertex.h"

namespace whiteice {

  class VisualizationInterface {
  public:

    // shows/opens graphics window
    virtual bool show() = 0;

    // hides/closes graphics window
    virtual bool hide() = 0;

    // gets maximum X+1 coordinate
    virtual unsigned int getScreenX() = 0;

    // gets maximum Y+1 coordinate
    virtual unsigned int getScreenY() = 0;

    // plots pixel to screen (white as the default)
    virtual bool plot(unsigned int x, unsigned int y,
		      unsigned int r = 255, unsigned int g = 255, unsigned int b = 255) = 0;

    // clears screen to wanted color (black as the default)
    virtual bool clear(unsigned int r = 0, unsigned int g = 0, unsigned int b = 0) = 0;

    // updates plotted pixels to screen
    virtual bool updateScreen() = 0;

    // blocks until key have been pressed
    virtual bool waitKeypress() = 0;

    // REAL FUNCTION: plots data points using virtual functions
    virtual bool adaptiveScatterPlot(const std::vector< math::vertex< math::blas_real<float> > >& points);
    
  };
  
  
};


#endif
