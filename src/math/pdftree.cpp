
#include <stdexcept>
#include <exception>
#include <vector>
#include <cmath>


#include "pdftree.h"


#ifndef pdftree_cpp
#define pdftree_cpp

namespace whiteice
{
  namespace math
  {
  
    // default contructor, use reset to init
    template <typename T, typename M>
    pdftree<T,M>::pdftree() throw()
    {
      root = 0;
      num_data = 0;
      split_limit = 30;
    }
    
    
    // constructor
    template <typename T, typename M>
    pdftree<T,M>::pdftree(const std::vector<T> min,
			  const std::vector<T> max) throw()
    {
      using namespace std;
      
      root = 0;
      num_data = 0;    
      this->min = min;
      this->max = max;
      
      // calculates volume of input space
      // especially don't use case: max == min (multip. with zero)      
      volume = T(1);    
      for(unsigned int i=0;i<min.size();i++)	
	if(max[i] > min[i])
	  volume *= (max[i]-min[i]);
      
      
      depth = 0;
      
      // each cell will contain ~ 8 items after split
      split_limit = (unsigned int)(8 * pow(2.0, (int)min.size()));
    }
    
    
    template <typename T, typename M>
    pdftree<T,M>::~pdftree() throw()
    {
      if(root) remove(root);
      root = 0;
    }
    
    
    /***************************************************/  
    
    // adds point to distribution
    template <typename T, typename M>
    bool pdftree<T,M>::add(const std::vector<T>& v) throw()
    {
      try{
	if(!v.size()) return false;
	if(v.size() != min.size()) return false;
	
	// checks data is in range [max, min]
	for(unsigned int i=0;i<min.size();i++)
	  if(v[i] > max[i] || v[i] < min[i]) return false;
	
	
	if(!root) root = new pdfnode<T>(min,max);
	depth_counter = 0;
	
	if(add(root,v)){
	  num_data++;
	  return true;
	}
	else return false;
      }
      catch(std::exception& e){
	return false;
      }
    }
    
    
    // calculates probability
    template <typename T, typename M>
    M pdftree<T,M>::pdf(const std::vector<T>& v) const throw()
    {
      if(!root) return M(1);
      depth_counter = 0;
      unsigned int i = pdf(root, v);
      
      // P(x E V_k) = Int(p(x), over V_k), assume p(x) constant in V_k
      // N/N_total  = p(x)V_k
      //       p(x) = N/(V_k*N_total)
      //            = N/(N_total*V_total/2^k)
      //            = (N*2^k)/(N_total*V_total)
      // divides area - assume equiprobable distribution
      
      M divide = pow((double)2, (double)depth_counter);
      
      return M( (M(i)*divide) / M(num_data * volume) );
    }
    
    // todo: calculates probability of area
    // bool probability_area(const std::vector<T>& v0,
    // const std::vector<T>& v1) const throw()
    //{
    //}
    
    
    // resets tree to a new tree
    template <typename T, typename M>
    bool pdftree<T,M>::reset(const std::vector<T>& min,
			     const std::vector<T>& max) throw()
    {
      using namespace std;
      
      try{
	num_data = 0;
	if(root) remove(root);
	root = 0;
	depth = 0;
	split_limit = (unsigned int)(8 * pow(2.0, (int)min.size()));
	
	this->min = min;
	this->max = max;
	
	// calculates volume of input space
	volume = T(1);    
	for(unsigned int i=0;i<min.size();i++)
	  if(max[i] > min[i]) // especially max - min != 0
	    volume *= (max[i]-min[i]);
	
	return true;
      }
      catch(std::exception& e){ return false; }    
    }
    
    
    // removes data points, not pdf information
    template <typename T, typename M>
    void pdftree<T,M>::flush() throw()
    {
      if(root) flush(root);
    }
    
    
    /**************************************************/
    // pdf tree handling
    
    
    // datapoint node or to subnode
    template <typename T, typename M>
    bool pdftree<T,M>::add(pdfnode<T>* node, const std::vector<T>& v) throw()
    {
      if(node->subtrees.size() <= 0){ // add to this node
	if(node->data.size() + 1 < split_limit){
	  node->data.push_back(v);
	  node->counter++;
	  if(depth_counter > depth) depth = depth_counter;
	  return true;
	}
	else{ // splits this node
	  
	  node->subtrees.resize(node->mid.size());
	  for(unsigned int i=0;i<node->mid.size();i++)
	    node->subtrees[i] = 0;
	  
	  // divides data to subtrees
	  unsigned int i=node->data.size();	
	  while(i > 0){
	    if(!add(node, node->data[i-1])) return false;
	    node->data.pop_back();
	    i = node->data.size();
	  }
	  
	  return add(node, v); // adds this node
	}
      }
      else{ // add to subnode
	
	// constructs index
	unsigned long index = 0;
	
	for(unsigned int i=0;i<node->mid.size();i++){
	  if(v[i] > node->mid[i])
	    index = (index << 1) + 1;
	  else
	    index = (index << 1) + 0;
	}
	
	
	if(!node->subtrees[index]){ // creates subnode
	  
	  // creates min/max values for subtree
	  std::vector<T> submin, submax;
	  unsigned int long t = index;
	  
	  for(unsigned int i=0;i<node->mid.size();i++){
	    if(t & 1){
	      submin.insert(submin.begin(), node->mid[i]);
	      submax.insert(submax.begin(), node->max[i]);
	    }
	    else{
	      submin.insert(submin.begin(), node->min[i]);
	      submax.insert(submax.begin(), node->mid[i]);
	    }
	    
	    t >>= 1;
	  }
	  
	  try{ node->subtrees[index] = new pdfnode<T>(submin, submax); }
	  catch(std::exception& e){ return false; }	
	}
	
	// adds data to subtree
	depth_counter++;
	if(add(node->subtrees[index], v)){
	  node->counter++; // also register counts at this level
	  depth_counter--;
	  return true;
	}
	else{
	  depth_counter--;
	  return false;
	}
      }
    }
    
    
    // returns number of elements in area E v and with depth = depth_counter
    template <typename T, typename M>
    unsigned int pdftree<T,M>::pdf(pdfnode<T>* node,
				   const std::vector<T>& v)
      const throw()
    {
      if(node->subtrees.size() <= 0)
	return node->counter;
      
      // calculates index to correct subtree
      unsigned long index = 0;
      
      for(unsigned int i=0;i<node->mid.size();i++){
	if(v[i] > node->mid[i]) index = (index << 1) + 1;
	else index = (index << 1);
      }
      
      if(!node->subtrees[index]){
	depth_counter++;
	return 0;
      }
      else{
	depth_counter++;
	return pdf(node->subtrees[index], v);
      }
    }
    
    
    template <typename T, typename M>
    void pdftree<T,M>::flush(pdfnode<T>* node) throw()
    {
      node->data.clear();
      
      for(unsigned int i=0;i<node->subtrees.size();i++)
	if(node->subtrees[i]) flush(node->subtrees[i]);
    }
    
    
    // frees this node and all subtrees
    template <typename T, typename M>
    void pdftree<T,M>::remove(pdfnode<T>* node) const throw()
    {
      if(!node) return;
      
      node->data.clear();
      
      typename std::vector< pdfnode<T>* >::iterator i;
      
      for(i=node->subtrees.begin();i!=node->subtrees.end();i++)
	if(*i) remove(*i);
      
      delete node;      
    }
    
    
    /***************************************************/
    // pdfnode constructor
    template <typename U>  
    pdfnode<U>::pdfnode(const std::vector<U>& min,
			const std::vector<U>& max) throw()
    {
      this->min = min;
      this->max = max;
      counter = 0;
      mid.resize(min.size());
      
      for(unsigned int i=0;i<min.size();i++)
	mid[i] = (U)(0.5*(min[i]+max[i]));      
    }
    
    
    
    
    template class pdftree<double, double>;
    template class pdftree<float, float>;
    template class pdfnode<double>;
    template class pdftree<float>;
    
  }
}
  


#endif

