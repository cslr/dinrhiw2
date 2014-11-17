/*
 * 2D SOM with wrap'a'round lattice distance
 * optimized cblas implementation 
 * (real floating point)
 */

#ifndef SOM2D_h
#define SOM2D_h

#include "vertex.h"
#include "data_source.h"
#include "ETA.h"


namespace whiteice
{
  
  class SOM2D
  {
  public:
    
    SOM2D(unsigned int width, unsigned int height,
	  unsigned int dimension);
    
    ~SOM2D();
    
    // learns given data, full is true performs also slow
    // convergence phase
    bool learn(whiteice::data_source< whiteice::math::vertex<float> >& datasource,
	       bool full=true) throw();
    
    // randomizes som vertex values
    bool randomize() throw();
    
    float somdistance(const whiteice::math::vertex<float>& v1,
		      const whiteice::math::vertex<float>& v2) const throw();
    
    // returns winner vertex raw index for a given vertex
    unsigned int activate(const whiteice::math::vertex<float>& v) const throw();
      
    // reads som vertex given lattice coordinate
    whiteice::math::vertex<float> operator()(unsigned int i, unsigned int j) const throw();
      
    // reads som vertex given direct raw index coordinate to a table
    whiteice::math::vertex<float> operator()(unsigned int index) const throw();    
    
    // size of the som lattice
    unsigned int width() const throw();
    unsigned int height() const throw();
    unsigned int dimension() const throw();
    unsigned int size() const throw();
    
    
    // loads SOM data from file
    bool load(const std::string& filename) throw();
    
    // saves SOM data to file
    bool save(const std::string& filename) const throw();
    
    // som visualization on/off switch
    bool show(bool on) throw();
    
  private:
    
    // finds the closest vector from som
    unsigned int find_winner(const float* vmemory) const throw();
    
    // calculates wrap-a-round distance for coordinates with given delta
    float wraparound_sqdistance(float dx, float dy) const throw();
    
    float norm_distance(float *m1, float* m2) const throw();

#if 0    
    bool open_visualization() throw();
    bool close_visualization() throw();
    bool draw_visualization() throw();
#endif
    
    
    // writes current ETA of SOM learning
    void report_eta(const unsigned int ITER,
		    const unsigned int MAXITER,
		    ETA<double>* eta);
    
    
    unsigned int som_width;
    unsigned int som_height;
    unsigned int som_dimension;

    float som_widthf;
    float som_heightf;
    float som_dimensionf;
    
    float hvariance0;
    float learning_rate0;
    float hvariance_t1;
    float learning_rate_t2;
    
    // width*height*dimension vectors
    float* somtable;
    
    float* umatrix; // for visualization
    
    bool show_visualization;
    bool show_eta;
    
    bool graphics_on;
  };
};


#endif



