//
//  threshold_finder.hpp
//  LinearGBMVector
//


//

#ifndef threshold_finder_hpp
#define threshold_finder_hpp

#include <vector>
#include "alignment_allocator.hpp"  

using std::vector;

class ThresholdFinder {
private:
    int num_threads;
public:
    ThresholdFinder(int num_threads); 
    
    double FindThreshold(const float *gradients, int top_k, int bins, int num_data);    
};

#endif /* threshold_finder_hpp */
