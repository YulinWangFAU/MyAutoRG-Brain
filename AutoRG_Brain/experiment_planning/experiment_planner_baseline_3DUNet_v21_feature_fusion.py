import os
import shutil

from batchgenerators.utilities.file_and_folder_operations import join

from configuration import default_num_threads
from experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21 as BasePlanner
from preprocess.preprocessing_llm_feature_fusion import GenericPreprocessor as FeatureFusionPreprocessor


class ExperimentPlanner3D_v21(BasePlanner):
    def run_preprocessing(self, num_threads, model_type="feature_fusion"):
        if os.path.isdir(join(self.preprocessed_output_folder, "gt_segmentations")):
            shutil.rmtree(join(self.preprocessed_output_folder, "gt_segmentations"))
        shutil.copytree(
            join(self.folder_with_cropped_data, "gt_segmentations"),
            join(self.preprocessed_output_folder, "gt_segmentations"),
        )

        normalization_schemes = self.plans["normalization_schemes"]
        use_nonzero_mask_for_normalization = self.plans["use_mask_for_norm"]
        intensityproperties = self.plans["dataset_properties"]["intensityproperties"]

        preprocessor = FeatureFusionPreprocessor(
            normalization_schemes,
            use_nonzero_mask_for_normalization,
            self.transpose_forward,
            intensityproperties,
        )

        target_spacings = [i["current_spacing"] for i in self.plans_per_stage.values()]
        if self.plans["num_stages"] == 1 and isinstance(num_threads, (list, tuple)):
            num_threads = num_threads[-1]
        elif self.plans["num_stages"] > 1 and not isinstance(num_threads, (list, tuple)):
            num_threads = (default_num_threads, num_threads)

        preprocessor.run(
            target_spacings,
            self.folder_with_cropped_data,
            self.preprocessed_output_folder,
            self.data_identifier,
            num_threads,
        )
