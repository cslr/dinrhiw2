/*
 * nonparameterized trivial probability distribution
 * modelling/counting with quad/oct/'n-dimensional quad' -trees
 *  - somewhat similar to parzen windows but with
 *    heuristical, adaptive h.
 * 
 *  - could also only keep counters of amount of data and
 *    when split happens divide data points equally between
 *    new areas. - error this causes diminishes when N->inf
 *    what about in practice?
 *
 *  - this is somewhat similar to polya trees
 *
 * RESTRICTION:
 *   supports maximum of 32 dimensions but easily
 *   generilizable (pdftree::add() index construction)
 *
 *   this isn't really meaningful restriction because
 *   pdftree scales only up to ~ 10 dimensions.
 *
 *  (with 32 variabes each node would split to 2^32
 *   subnodes. this is very unpractical)
 *
 */

#include <vector>

#ifndef pdftree_h
#define pdftree_h

namespace whiteice
{
  namespace math
  {
    
    template <typename U>
      class pdfnode;
    
    
    /*
     * T is data type for input data
     * M is probability distribution calculation type
     */
    template <typename T=double, typename M=double>
      class pdftree
    {    
      public:
      
      pdftree() throw();
      pdftree(const std::vector<T> min,
	      const std::vector<T> max) throw();
      
      ~pdftree() throw();
      
      // adds point to distribution
      bool add(const std::vector<T>& v) throw();
      
      // calculates probability
      M pdf(const std::vector<T>& v) const throw();
      
      // todo: xcalculates probability of area
      //bool probability_area(const std::vector<T>& v0, const std::vector<T>& v1) const throw();
      
      // resets tree to a new tree
      bool reset(const std::vector<T>& min,
		 const std::vector<T>& max) throw();
      
      // removes all data information
      //   - frees memory, pdf data is still available
      //   - after this one creation of subtrees
      //     start with 'fresh'/no memory
      void flush() throw();
      
      /***************************************************/
      private:            
      
      
      // adds data to node or to subnode
      bool add(pdfnode<T>* node, const std::vector<T>& v) throw();
      
      // returns number of elements in area E v and with depth = depth_counter
      unsigned int pdf(pdfnode<T>* node, const std::vector<T>& v) const throw();
      
      // removes data points
      void flush(pdfnode<T>* node) throw();
      
      // frees subnodes and node itself
      void remove(pdfnode<T>* node) const throw();
      
      
      unsigned int num_data;
      std::vector<T> min, max;
      T volume;
      
      // used by add/pdf calc for updating depth
      mutable unsigned int depth_counter;
      
      
      pdfnode<T> *root;
      unsigned int depth; // depth of the tree
      
      unsigned int split_limit; // splits node when amount of data > split_limit
    };
    
    
    // pdfnode class for pdftree
    template <typename U>
      class pdfnode
      {
      public:
	pdfnode(const std::vector<U>& min,
		const std::vector<U>& max) throw();
	
	std::vector<U> min, mid, max;
	unsigned int counter;
	std::vector< std::vector<U> > data;
	std::vector< pdfnode<U>* > subtrees;
      };
   
    
    
    
    extern template class pdftree<double, double>;
    extern template class pdftree<float, float>;
    extern template class pdfnode<double>;
    extern template class pdftree<float>;
    
  }      
}


  
#endif




