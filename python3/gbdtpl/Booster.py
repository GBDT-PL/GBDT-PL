from ctypes import *
import numpy as np
import os

class Booster:
    def __init__(self, params, train_data, test_data):
        cwd = os.path.dirname(os.path.abspath(__file__))	
        self.LinearGBMLib = cdll.LoadLibrary(cwd + "/liblineargbm.py")
        self.booster_config = c_void_p()
        self.params = params
        self.LinearGBMLib.CreateLinearGBMBoosterConfig(byref(self.booster_config))
        for key in params:
            self.LinearGBMLib.SetLinearGBMParams(self.booster_config, c_char_p(key.encode("utf-8")),
                                            c_char_p(str(params[key]).encode("utf-8")))
        self.booster = c_void_p()
        self.LinearGBMLib.CreateLinearGBM(self.booster_config, train_data.data_mat, test_data.data_mat, byref(self.booster))

    def Train(self):
         self.LinearGBMLib.Train(self.booster)

    def __c_pointer_to_numpy_array(self, preds, num_data):
        numpy_preds = np.zeros(num_data, dtype=np.float64)
        memmove(numpy_preds.ctypes.data, preds, num_data * numpy_preds.strides[0])
        return numpy_preds

    def Predict(self, test_data, iters=-1):
        preds = POINTER(c_double)()
        num_data = c_int()
        self.LinearGBMLib.LinearGBMPredict(self.booster, test_data.data_mat, byref(preds), byref(num_data), c_int(iters))
        return self.__c_pointer_to_numpy_array(preds, num_data.value)

    def get_best_iteration(self):
        best_iteration = c_int()
        best_score = c_double()
        self.LinearGBMLib.LinearGBMBestIteration(self.booster, byref(best_iteration), byref(best_score))
        return best_iteration.value, best_score.value

    def get_scores_per_iteration(self, data_name):
        scores = POINTER(c_double)()
        self.LinearGBMLib.LinearGBMGetScoresPerIteration(self.booster, c_char_p(data_name.encode("utf-8")), byref(scores))
        return self.__c_pointer_to_numpy_array(scores, self.params["num_trees"])


