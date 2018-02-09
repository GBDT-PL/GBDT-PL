//
//  threshold_finder.cpp
//  LinearGBMVector
//


//

#include "threshold_finder.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

using std::cout;
using std::endl;

ThresholdFinder::ThresholdFinder(int num_threadss) { num_threads = num_threadss; }

double ThresholdFinder::FindThreshold(const float *gradients, int top_k, int bins, int num_data) { 
    vector<float> boundaries(bins + 1, 0.0);
    vector<vector<int>> thread_counts(num_threads);
    for(int i = 0; i < num_threads; ++i) {
        thread_counts[i].resize(bins, 0);
    }
    
    
    vector<double> thread_upper_bound(num_threads, 0.0);
    vector<double> thread_lower_bound(num_threads, 1e6);
    int chunk_size = (num_data + num_threads - 1) / num_threads;        
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for(int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, num_data);
        for(int j = start; j < end; ++j) {
            double abs = fabs(gradients[j * 2] * gradients[j * 2 + 1]);
            if(abs > thread_upper_bound[i]) {
                thread_upper_bound[i] = abs;
            }
            if(abs < thread_lower_bound[i]) {
                thread_lower_bound[i] = abs;
            }
        }
    }
    
    double upper_bound = 0.0, lower_bound = 1e6;
    for(int i = 0; i < num_threads; ++i) {
        if(upper_bound < thread_upper_bound[i]) {
            upper_bound = thread_upper_bound[i];
        }
        if(lower_bound > thread_lower_bound[i]) {
            lower_bound = thread_lower_bound[i];    
        }
    }
    for(int i = 0; i < bins + 1; ++i) {
        boundaries[i] = (i * upper_bound + (bins - i) * lower_bound) / bins;
    }
    boundaries[0] -= 1e-3;
    boundaries.back() += 1e-3;
    
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for(int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, num_data);
        for(int j = start; j < end; ++j) {
            int pos = static_cast<int>(std::lower_bound(boundaries.begin(), boundaries.end(),
                                                        fabs(gradients[j * 2] * gradients[j * 2 + 1])) - boundaries.begin());
            assert(pos >= 1);
            ++thread_counts[i][pos - 1];
        }
    }
    
    for(int i = 1; i < num_threads; ++i) {
#pragma omp simd
        for(int j = 0; j < bins; ++j) {
            thread_counts[0][j] += thread_counts[i][j]; 
        }
    }
    
    int cur_count = 0;
    for(int i = bins - 1; i >= 0; --i) {
        cur_count += thread_counts[0][i];
        if(cur_count >= top_k) {
            cout << "sampling " << cur_count << endl;
            return boundaries[i];   
        }
    }
    return 0.0; 
}
