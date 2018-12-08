//
//  data_reader.cpp
//  LinearGBM
//


//

#include "data_reader.hpp"
#include <iostream>
#include <cassert>
#include <algorithm>

using std::cout;
using std::endl;

DataReader::DataReader(string file_path_, int buffer_size_,
                       vector<vector<double>>& feature_values_, vector<double> &label_, int label_idx_):
feature_values(feature_values_), label(label_) {
    num_features = 0;
    num_data = 0;
    file_path = file_path_;
    buffer_size = buffer_size_;
    label_idx = label_idx_;
}

void DataReader::Read(int num_threads) {
    if(num_threads == 1) {
    	num_threads = 2;
    }
    ifstream fin_line_count(file_path);
    string line_count;
    num_data = 1;
    num_features = 1;
    getline(fin_line_count, line_count);
    for(int i = 0; i < line_count.size(); ++i) {
        if(line_count[i] == ',') {
            ++num_features;
        }
    }
    //subtract label
    --num_features;
    while(getline(fin_line_count, line_count)) {
        ++num_data;
    }
    feature_values.resize(num_features);
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for(int i = 0; i < num_features; ++i) {
        feature_values[i].resize(num_data, 0.0);
    }
    label.resize(num_data, 0.0);
    fin_line_count.close();
    
    ifstream fin(file_path);
    string line;
    while(getline(fin, line)) {
        buffer1.emplace_back(line.c_str());
        if(buffer1.size() == buffer_size) {
            break;
        }
    }
    vector<string>* buffer_ptr_1 = &buffer1;
    vector<string>* buffer_ptr_2 = &buffer2;
    
    size_t cur_total_lines = 0;
    while(cur_total_lines < num_data) { 
        size_t chunk_size = (buffer_ptr_1->size() + num_threads - 2) / (num_threads - 1); 
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int tid = 0; tid < num_threads; ++tid) {
            if(tid == 0) {
                for(int i = 0; i < buffer_size && getline(fin, line); ++i) {
                    buffer_ptr_2->emplace_back(line.c_str());
                }
            }
            else {
                size_t start = chunk_size * (tid - 1);
                size_t end = std::min(start + chunk_size, buffer_ptr_1->size());
                vector<string> &buffer = *buffer_ptr_1;
                double value = 0.0;
                for(size_t i = start; i < end; ++i) {
                    int fid = 0;
                    int begin = 0;
                    const char* c_str = buffer[i].c_str();
                    for(int j = 0; j < buffer[i].size(); ++j) {
                        if(buffer[i][j] == ',') {
                            value = std::atof(c_str + begin);

                            if(fid >= 0 && fid < label_idx) {
                                feature_values[fid][cur_total_lines + i] = value;
                            }
                            else if(fid == label_idx) {	
                                label[cur_total_lines + i] = value;
                            }
			    else {
			    	feature_values[fid - 1][cur_total_lines + i] = value;
			    }
                            begin = j + 1;
                            ++fid;
                        }
                    }
                    value = std::atof(c_str + begin);
                    if(fid >= 0 && fid < label_idx) {
                        feature_values[fid][cur_total_lines + i] = value;
                    }
                    else if(fid == label_idx) {
                        label[cur_total_lines + i] = value;
                    }
		    else {
		        feature_values[fid - 1][cur_total_lines + i] = value;
		    }
                }
            }
        }
        cur_total_lines += buffer_ptr_1->size();
        buffer_ptr_1->clear();
        std::swap(buffer_ptr_1, buffer_ptr_2);
    }
}

void DataReader::ReadByRow(int num_threads) {
    if(num_threads == 1) {
    	num_threads = 2;
    }
    ifstream fin_line_count(file_path);
    string line_count;
    num_data = 1;
    num_features = 1;
    getline(fin_line_count, line_count);
    for(int i = 0; i < line_count.size(); ++i) {
        if(line_count[i] == ',') {
            ++num_features;
        }
    }
    //subtract label
    --num_features;
    while(getline(fin_line_count, line_count)) {
        ++num_data;
    }
    feature_values.resize(num_data);
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for(int i = 0; i < num_data; ++i) {
        feature_values[i].resize(num_features, 0.0);
    }
    label.resize(num_data, 0.0);
    fin_line_count.close(); 

    ifstream fin(file_path);
    string line;
    while(getline(fin, line)) {
        buffer1.emplace_back(line.c_str());
        if(buffer1.size() == buffer_size) {
            break;
        }
    }
    vector<string>* buffer_ptr_1 = &buffer1;
    vector<string>* buffer_ptr_2 = &buffer2;
 
    size_t cur_total_lines = 0;
    while(cur_total_lines < num_data) {
        size_t chunk_size = (buffer_ptr_1->size() + num_threads - 2) / (num_threads - 1);
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int tid = 0; tid < num_threads; ++tid) {
            if(tid == 0) {
                for(int i = 0; i < buffer_size && getline(fin, line); ++i) {
                    buffer_ptr_2->emplace_back(line.c_str());
                }
            }
            else {
                size_t start = chunk_size * (tid - 1);
                size_t end = std::min(start + chunk_size, buffer_ptr_1->size());
                vector<string> &buffer = *buffer_ptr_1;
                double value = 0.0;
                for(size_t i = start; i < end; ++i) {
                    int fid = 0;
                    int begin = 0;
                    const char* c_str = buffer[i].c_str();
                    for(int j = 0; j < buffer[i].size(); ++j) {
                        if(buffer[i][j] == ',') {
                            value = std::atof(c_str + begin);

                            if(fid >= 0 && fid < label_idx) {
                                feature_values[cur_total_lines + i][fid] = value;
                            }
                            else if(fid == label_idx) {
                                label[cur_total_lines + i] = value;
                            }
			    else {
			        feature_values[cur_total_lines + i][fid - 1] = value;
			    }
                            begin = j + 1;
                            ++fid;
                        }
                    }
                    value = std::atof(c_str + begin);
                    if(fid >= 0 && fid < label_idx) {
                        feature_values[cur_total_lines + i][fid] = value;
                    }
                    else if(fid == label_idx) {
                        label[cur_total_lines + i] = value;
                    }
		    else {
		        feature_values[cur_total_lines + i][fid - 1] = value;
		    }
                }
            }
        }
        cur_total_lines += buffer_ptr_1->size();
        buffer_ptr_1->clear();
        std::swap(buffer_ptr_1, buffer_ptr_2);
    } 
}
