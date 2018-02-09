//
//  argmax.h
//  LinearGBMVector
//


//

#ifndef argmax_h
#define argmax_h

#include <cmath>
#include <functional>
#include <vector>

using std::function;
using std::vector;

template <typename T>
T* ArgMax(int low, int high, vector<T*> &data, int k, function<bool(T*,T*)> comp);

#endif /* argmax_h */
