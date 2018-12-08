from ctypes import *
import numpy as np

class Booster:
    def __init__(self, params, train_data, test_data):
        self.LinearGBMLib = cdll.LoadLibrary("../lib/liblineargbm.so")
        self.booster_config = c_void_p()
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

