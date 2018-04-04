from ctypes import *

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
