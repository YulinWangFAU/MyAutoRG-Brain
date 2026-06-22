import os
import pickle

import numpy as np

from preprocess.preprocessing_llm import GenericPreprocessor as BaseGenericPreprocessor


class GenericPreprocessor(BaseGenericPreprocessor):
    @staticmethod
    def load_cropped(cropped_output_dir, case_identifier):
        all_data = np.load(os.path.join(cropped_output_dir, "%s.npz" % case_identifier))["data"]
        data = all_data[:4].astype(np.float32)
        seg = all_data[4:]
        with open(os.path.join(cropped_output_dir, "%s.pkl" % case_identifier), "rb") as f:
            properties = pickle.load(f)
        return data, seg, properties
