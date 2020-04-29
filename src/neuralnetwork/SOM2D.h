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
#include "RNG.h"

#include <vector>
#include <list>


namespace whiteice
{

  class SOM2D
  {
  public:
    
    SOM2D(unsigned int width, unsigned int height,
	  unsigned int dimension);

    SOM2D(const SOM2D& som);
    
    ~SOM2D();
    
    // learns given data, full is true performs also slow
    // convergence phase
    bool learn(const std::vector< whiteice::math::vertex< whiteice::math::blas_real<float> > >& datasource,
	       bool full=true) throw();
    
    // randomizes som vertex values
    bool randomize() throw();
    
    // randomsizes SOM vertex values to span two highest variance PCA eigenvectors
    bool randomize 
    (const std::vector< whiteice::math::vertex< whiteice::math::blas_real<float> > >& data) throw();

    // uses smaller SOM to initialize weights of this SOM
    bool initializeHiearchical(const SOM2D& som_prev);
    

    // calculates average error by using the best match vector for given dataset's vectors
    whiteice::math::blas_real<float> getError(const std::vector< whiteice::math::vertex< whiteice::math::blas_real<float> > >& data);
    
    float somdistance(const whiteice::math::vertex< whiteice::math::blas_real<float> >& v1,
		      const whiteice::math::vertex< whiteice::math::blas_real<float> >& v2) const throw();
    
    // returns winner vertex raw index for a given vertex
    unsigned int activate(const whiteice::math::vertex< whiteice::math::blas_real<float> >& v) const throw();

    // returns interpolated coordinates in SOM 2d map:
    // first find winner vertex (i,j) and calculates its closeness to data at points
    // at (i-1,j), (i+1,j), (i,j-1), (i,j+1). weight of each location is |x^t * |som(i,j)|
    bool smooth_coordinate(const whiteice::math::vertex< whiteice::math::blas_real<float> >& v,
			   whiteice::math::vertex< whiteice::math::blas_real<float> > smooth_coordinate);

    // assigns pseudo probability value to data v based on vector in SOM2D(i,j)
    whiteice::math::blas_real<float> getActivity
    (const whiteice::math::vertex< whiteice::math::blas_real<float> >& v,
     unsigned int i, unsigned int j) const throw();
      
    // reads som vertex given lattice coordinate
    whiteice::math::vertex< whiteice::math::blas_real<float> > operator()(int i, int j) const throw();

    // writes vertex given lattice coordinate
    bool setVector(int i, int j,
		   const whiteice::math::vertex< whiteice::math::blas_real<float> >& v);
      
    // reads som vertex given direct raw index coordinate to a table
    whiteice::math::vertex< whiteice::math::blas_real<float> > operator()(unsigned int index) const throw();

    // handles wrap-around property properly, returns index to n:th vector, for RAW
    // index value multiply by som_dimension..
    bool index2coordinates(const unsigned int index, unsigned int& i, unsigned int& j) const throw();
    bool coordinates2index(const unsigned int i, const unsigned int j,
			   unsigned int& index) const throw();

    whiteice::math::blas_real<float> Uvalue() const throw();
    
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
    unsigned int find_winner(const whiteice::math::blas_real<float>* vmemory) const throw();
    
    // calculates wrap-a-round distance for coordinates with given delta
    float wraparound_sqdistance(float x1, float x2, float y1, float y2) const throw();
    
    float norm_distance(const whiteice::math::blas_real<float> * const m1,
			const whiteice::math::blas_real<float> * const m2) const throw();

#if 0    
    bool open_visualization() throw();
    bool close_visualization() throw();
    bool draw_visualization() throw();
#endif
    
    
    // writes current ETA of SOM learning
    void report_convergence
    (const unsigned int ITER,
     const unsigned int MAXITER,
     std::list< whiteice::math::blas_real<float> >& errors,
     ETA<double>* eta,
     const std::vector< whiteice::math::vertex< whiteice::math::blas_real<float> > >& datasource);
    
    
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
    whiteice::math::blas_real<float>* somtable;
    
    whiteice::math::blas_real<float>* umatrix; // for visualization
    
    bool show_visualization;
    bool show_eta;
    
    bool graphics_on;

    whiteice::RNG< whiteice::math::blas_real<float> > rng;
  };
};


#endif



