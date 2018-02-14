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
|  HEPMASS   | 7,000,000 | 3500000 | 28 | Classification | [hepmass](https://archive.ics.uci.edu/ml/datasets/HEPMASS)|
| CASP | 30,000 | 15,730 | 9 | Regression | [casp](https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure) |
| Poly | 2,000,000 | 1,000,000 | 200 | Regression | TBA |
| Cubic | 10,000,000 | 1,000,000 | 10 | Regression | TBA |

For Higgs, we use the first 10,000,000 data points as training set, and the reset as testing set. For Epsilon, we use the first 400,000 as training and the reset as testing. For HEPMASS, we use the first 7,000,000 as training and the reset as testing. For CASP, we use the first 30,000 as training and the reset as testing. 

### Convergence Rate
We run 500 iterations and plot testing accuracy per iteration. Figure 4 shows the results. We use lgb for [LightGBM](https://github.com/Microsoft/LightGBM) and xgb for [XGBoost](https://github.com/dmlc/xgboost) in the figure legends. 


