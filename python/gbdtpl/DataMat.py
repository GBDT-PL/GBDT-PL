from ctypes import *

class DataMat:
    def __init__(self, name, params, label_index, query_index, file_path, reference=None):
        self.LinearGBMLib = cdll.LoadLibrary("../lib/liblineargbm.so")
        self.booster_config = c_void_p()
        self.LinearGBMLib.CreateLinearGBMBoosterConfig(byref(self.booster_config))
        for key in params:
            self.LinearGBMLib.SetLinearGBMParams(self.booster_config, c_char_p(key.encode("utf-8")), c_char_p(str(params[key]).encode("utf-8")))
        self.LinearGBMLib.LinearGBMPrintBoosterConfig(self.booster_config)
        self.data_mat = c_void_p()
        if reference is None:
            self.LinearGBMLib.CreateLinearGBMDataMat(self.booster_config, c_char_p(name.encode("utf-8")),
                                            c_int(label_index), c_int(query_index), c_char_p(file_path.encode("utf-8")), byref(self.data_mat), c_void_p(0))
        else: 
            self.LinearGBMLib.CreateLinearGBMDataMat(self.booster_config, c_char_p(name.encode("utf-8")),
                                                c_int(label_index), c_int(query_index), c_char_p(file_path.encode("utf-8")), byref(self.data_mat), reference.data_mat)
