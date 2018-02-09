//
//  argmax.cpp
//  LinearGBMVector
//


//

#include <stdio.h>
#include "argmax.h"

template <typename T>
int partition(int low, int high, vector<T*> &data, int k, function<bool(T*,T*)> comp) {
    int l = low, h = high;
    T* pivot = data[h];
    while(l < h) {
        while(comp(data[l], pivot)) {
            ++l;
        }
        while(comp(pivot, data[h])) {
            --h;
        }
        if(!comp(data[l], data[h]) && !(comp(data[h], data[l]))) {
            ++l;
        }
        else if(l < h) {
            T* tmp = data[l];
            data[l] = data[h];
            data[h] = tmp;
        }
    }
    return h;
}

template <typename T>
T* qsort_select(int low, int high, vector<T*> &data, int k, function<bool(T*,T*)> comp) {
    if(low == high) {
        return data[low];
    }
    
    int mid = partition(low, high, data, k, comp);
    if(mid - low + 1 == k) {
        return data[mid];
    }
    else if(mid - low + 1 > k) {
        return qsort_select(low, mid - 1, data, k, comp);
    }
    else {
        return qsort_select(mid + 1, high, data, k - (mid - low + 1), comp);
    }
}

template <typename T>
T* ArgMax(int low, int high, vector<T*> &data, int k, function<bool(T*,T*)> comp) {
    return qsort_select(low, high, data, k, comp);                                                  
}

