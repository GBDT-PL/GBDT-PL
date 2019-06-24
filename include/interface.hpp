#ifndef interface_h
#define interface_h

typedef void* LinearGBM;
typedef void* LinearGBMDataMat;
typedef void* LinearGBMBoosterConfig;

extern "C" int CreateLinearGBM(LinearGBMBoosterConfig booster_config,
                               LinearGBMDataMat train_data,
                               LinearGBMDataMat test_data,
                               LinearGBM *out);

extern "C" int CreateLinearGBMBoosterConfig(LinearGBMBoosterConfig *out); 

extern "C" int SetLinearGBMParams(LinearGBMBoosterConfig booster_config,    
                                  const char* key, const char* value);

extern "C" int CreateLinearGBMDataMat(LinearGBMBoosterConfig booster_config, const char* name,
                                      int label_index, int query_index,
                                      const char* file_path, LinearGBMDataMat *out,
                                      LinearGBMDataMat reference = nullptr);

extern "C" int Train(LinearGBM gbm);

extern "C" int LinearGBMPrintBoosterConfig(LinearGBMBoosterConfig booster_config);

extern "C" int LinearGBMPredict(LinearGBM booster, LinearGBMDataMat test_data, double** preds, int* num_data, int iters);

extern "C" int LinearGBMBestIteration(LinearGBM booster, int* best_iteration, double* best_score);

extern "C" int LinearGBMGetScoresPerIteration(LinearGBM booster, const char* name, double** scores);

#endif /* interface_h */
