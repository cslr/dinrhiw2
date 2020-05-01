/*
 * augmented data is data structure with
 * key value (comparisions) with other data
 * mostly for tree etc. structures
 *
 * K - key data type
 * D - extra data type
 *
 * inlined - try to cause minimum impact on performance
 */

#ifndef augmented_data_h
#define augmented_data_h

#include <iostream>


namespace whiteice
{
  
  template <typename K, typename D>
    class augmented_data
    {
    public:
      inline augmented_data(){
	keyValue = K(); dataValue = D();
      }
      
      inline augmented_data(const augmented_data<K,D>& a){
	keyValue = a.keyValue; dataValue = a.dataValue;
      }
      
      inline augmented_data(const K& key_, const D& data_){
	keyValue = key_; dataValue = data_;
      }
      
      inline K& key() { return keyValue; }
      inline const K& key() const { return keyValue; }
      
      inline D& data() { return dataValue; }
      inline const D& data() const { return dataValue; }
      
      inline augmented_data<K,D>& operator=(const augmented_data<K,D>& d) {
	keyValue = d.keyValue; dataValue = d.dataValue;
	return *this;
      }
      
      inline bool operator==(const augmented_data<K,D>& d) const {
	return (keyValue == d.keyValue);
      }
      
      inline bool operator!=(const augmented_data<K,D>& d) const {
	return (keyValue != d.keyValue);
      }
      
      inline bool operator> (const augmented_data<K,D>& d) const {
	return (keyValue >  d.keyValue);
      }
      
      inline bool operator>=(const augmented_data<K,D>& d) const {
	return (keyValue >= d.keyValue);
      }
      
      inline bool operator< (const augmented_data<K,D>& d) const {
	return (keyValue <  d.keyValue);
      }
      
      inline bool operator<=(const augmented_data<K,D>& d) const {
	return (keyValue <= d.keyValue);
      }
      
    private:
      K keyValue;
      D dataValue;
    };
  
  
  template <typename K, typename D>
    std::ostream& operator<<(std::ostream& ios,
			     const whiteice::augmented_data<K,D>& d)
    {
      ios << d.key();
      return ios;
    }
  
}
    
#endif



