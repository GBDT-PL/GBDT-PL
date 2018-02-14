# GBDT-PL
We extend gradient boosting to use piecewise linear regression trees (PL Trees), 
instead of piecewise constant regression trees. PL Trees can accelerate convergence of
GBDT. Moreover, our new algorithm fits better to modern computer architectures with powerful
Single Instruction Multiple Data (SIMD) parallelism. We name our new algorithm GBDT-PL.

## Experiments 
We evaluate our algorithm on 4 public datasets. In addition, we create 2 synthetic datasets. 

|Dataset Name| #Training | #Testing | #Features |      Task      | Link |
|------------|------------|----------|-----------|----------------|------|
|    Higgs   | 10,000,000 | 500,000  |     28    | Classification | [higgs](https://archive.ics.uci.edu/ml/datasets/HIGGS) |
|   Epsilon  | 400,000 | 100,000  |     2000    | Classification | [epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) |
