/*
 * neural network implementation (V2)
 * work arounds some bugs + has more efficient implementation
 * 
 */

#ifndef nnetwork_h
#define nnetwork_h


#include "atlas.h"
#include "vertex.h"
#include "conffile.h"
#include "compressable.h"
#include "MemoryCompressor.h"
#include <vector>


namespace whiteice
{
  
  template < typename T = math::atlas_real<float> >
    class nnetwork
    {
    public:
    
    // creates useless 1x1 network. 
    // Use load to load some useful network
    nnetwork(); 
    nnetwork(const nnetwork& nn);
    nnetwork(const std::vector<unsigned int>& nnarch) throw(std::invalid_argument);
    
    virtual ~nnetwork();

    nnetwork<T>& operator=(const nnetwork<T>& nn);
    
    ////////////////////////////////////////////////////////////
    
    math::vertex<T>& input() throw(){ return inputValues; }
    math::vertex<T>& output() throw(){ return outputValues; }
    const math::vertex<T>& input() const throw(){ return inputValues; }
    const math::vertex<T>& output() const throw(){ return outputValues; }
    
    // returns input and output dimensions of neural network
    unsigned int input_size() const throw();
    unsigned int output_size() const throw();

    void getArchitecture(std::vector<unsigned int>& arch) const;
    
    bool calculate(bool gradInfo = false);
    bool operator()(bool gradInfo = false){ return calculate(gradInfo); }
    
    unsigned int length() const; // number of layers
    
    bool randomize();
    
    // calculates gradient grad(error) = grad(right - output)
    bool gradient(const math::vertex<T>& error,
		  math::vertex<T>& grad) const;
    
    ////////////////////////////////////////////////////////////
    
    // load & saves neuralnetwork data from file
    bool load(const std::string& filename) throw();
    bool save(const std::string& filename) const throw();
    
    // exports and imports neural network parameters to/from vertex
    bool exportdata(math::vertex<T>& v) const throw();
    bool importdata(const math::vertex<T>& v) throw();
    
    // number of dimensions used by import/export
    unsigned int exportdatasize() const throw(); 
    
    // changes NN to compressed form of operation or
    // back to normal non-compressed form
    
    
    ////////////////////////////////////////////////////////////
    private:
    
    inline void gemv(unsigned int yd, unsigned int xd, T* W, T* x, T* y);
    inline void gvadd(unsigned int dim, T* s, T* b);

    
    inline void gemv_gvadd(unsigned int yd, unsigned int xd, 
			   T* W, T* x, T* y,
			   unsigned int dim, T* s, T* b);
    
    
    // data structures which are part of
    // interface
    mutable math::vertex<T> inputValues;
    mutable math::vertex<T> outputValues;
    

    bool hasValidBPData;
    
    // architecture (eg. 3-2-6) info
    std::vector<unsigned int> arch;
    unsigned int maxwidth;    
    unsigned int size;

    std::vector<T> data;
    std::vector<T> bpdata;
    
    std::vector<T> state;
    std::vector<T> temp;

    // bool compressed;
    // MemoryCompressor* compressor;
  };
  
  
  
  extern template class nnetwork< float >;
  extern template class nnetwork< double >;  
  extern template class nnetwork< math::atlas_real<float> >;
  extern template class nnetwork< math::atlas_real<double> >;
  
};


#endif

