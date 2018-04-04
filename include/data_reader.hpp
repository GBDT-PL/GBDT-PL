//
//  file_reader.hpp
//  LinearGBM
//


//

#ifndef file_reader_h
#define file_reader_h

#include <string>
#include <vector>
#include <fstream>
#include <omp.h>

using std::string;
using std::vector;
using std::ifstream;
using std::getline;

class DataReader {
private:
    string file_path;
    vector<string> buffer1, buffer2;
    vector<vector<double>> &feature_values;
    vector<double> &label;
    int buffer_size;
    int num_features;
    int num_data;
    
public:
    DataReader(string file_path_, int buffer_size_,
               vector<vector<double>> &feature_values_, vector<double> &label_); 
    
    void Read(int num_threads);
    void ReadByRow(int num_threads);
    
    int get_num_features() { return num_features; }
    int get_num_data() { return num_data; }
};

#endif /* file_reader_h */

