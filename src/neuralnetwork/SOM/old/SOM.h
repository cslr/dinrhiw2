/*
 * simple 2d SOM
 * TODO: implementation could be much better..
 *  many passes for SOM, probability distribution modification
 *  (using histrogram modification to wanted distribution)
 *  to correct SOMs f^2/3 or f^/1/3 error in probability vs
 *  number of representing vectors in SOM.
 *  learning should happen in two/many passes etc. (Haykin's book)
 *  etc. etc.
 */

#ifndef SOM_h
#define SOM_h

#include <vector>
#include <string>
#include <exception>
#include <stdexcept>

namespace whiteice
{

  template <typename T>
    class SOM
    {
    public:
      
      // creates 2d som lattice with given height, width and data dimension
      // width and height must be powers of 2, otherwise exception is throwed
      SOM(unsigned int width = 2, unsigned int height = 2, unsigned int dimension = 1)
	throw(std::logic_error);
      
      virtual ~SOM();
      
      // run SOM with given data
      bool learn(const std::vector< std::vector<T> >& data) throw();
      
      // returns distance in feature space
      T som_distance(const std::vector<T>& v,
		 const std::vector<T>& w) const throw();
      
      // returns index to vector representing given vector
      unsigned int representing_vector(const std::vector<T>& v) const throw();
      
      // returns som vector for given vector index
      std::vector<T>& operator[](unsigned int index) const throw(std::out_of_range);
      
      // randomizes som values
      bool randomize() const throw();
      
      // loads SOM data from file
      bool load(const std::string& filename) throw();
      
      // saves SOM data to file
      bool save(const std::string& filename) const throw();
      
      // returns number of som vectors
      unsigned int size() const throw(){ return som.size(); }
      
    private:
      
      // finds closest som representation
      // vector index for given vector
      unsigned int find_closest(const std::vector<T>& data)
	const throw();
      
      
      // normalizes length of the vector to 1
      // or keeps it zero if it's zero
      void normalize_length(std::vector<T>& v) const throw();
      
      // calculates squared distance between vectors in som lattice
      T som_sqr_distance(unsigned int index1,
			 unsigned int index2) const throw();
      
      // calculates |x - y|^2
      T vector_sqr_distance(const std::vector<T>& x,
			    const std::vector<T>& y) const throw()  ;
      
      // moves all vectors in winners neighbourhood
      // towards data vector
      bool move_towards(unsigned int winner,
			const std::vector<T>& data) throw();
      
      // finds closests som representation
      // vector index for given vector
      unsigned int find_closests(const std::vector<T>& data)
	const throw();
      
      // open()s graphic display/window
      bool  open_visualization() throw();
      bool close_visualization() throw();
      bool  draw_visualization() throw();
      
      
      std::vector< std::vector<T> > som;
      std::vector<T> umatrix;
      
      unsigned int wbits; // 2^wbits = width
      unsigned int width, height;
      unsigned int dimension;
      
      float initial_learning_rate;
      float initial_variance_distance;
      float target_variance_distance;
      
      float learning_rate;
      float variance_distance;
      
      bool graphics_on;
      
      bool show_visualization;
      bool show_eta;            
    };
  
}
  
#include "SOM.cpp"


#endif






