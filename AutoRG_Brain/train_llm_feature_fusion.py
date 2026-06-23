import importlib
import os
import runpy
import sys
import types


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

nnunet_alias = types.ModuleType("nnunet")
nnunet_alias.__path__ = [os.path.dirname(os.path.abspath(__file__))]
sys.modules.setdefault("nnunet", nnunet_alias)
sys.modules.setdefault("nnunet.paths", importlib.import_module("paths"))

fused_lm = importlib.import_module("network.language_model_patchwise_fused")
sys.modules["network.language_model_patchwise"] = fused_lm

feature_trainer = importlib.import_module("network_training.nnUNetTrainerV2_llm_feature_fusion")
sys.modules["network_training.nnUNetTrainerV2_llm_resize_multi"] = feature_trainer

print("[TRAIN FEATURE FUSION] language_model_patchwise -> language_model_patchwise_fused")
print("[TRAIN FEATURE FUSION] nnUNetTrainerV2_llm_resize_multi -> nnUNetTrainerV2_llm_feature_fusion")

runpy.run_module("train_llm_multi", run_name="__main__")
