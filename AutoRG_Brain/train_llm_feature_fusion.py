import importlib
import runpy
import sys


fused_lm = importlib.import_module("network.language_model_patchwise_fused")
sys.modules["network.language_model_patchwise"] = fused_lm

feature_trainer = importlib.import_module("network_training.nnUNetTrainerV2_llm_feature_fusion")
sys.modules["network_training.nnUNetTrainerV2_llm_resize_multi"] = feature_trainer

print("[TRAIN FEATURE FUSION] language_model_patchwise -> language_model_patchwise_fused")
print("[TRAIN FEATURE FUSION] nnUNetTrainerV2_llm_resize_multi -> nnUNetTrainerV2_llm_feature_fusion")

runpy.run_module("train_llm_multi", run_name="__main__")
