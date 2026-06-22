from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from utilities.task_name_id_conversion import convert_id_to_task_name
from paths import *
from preprocess.sanity_checks_llm import verify_dataset_integrity
from experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from experiment_planning.utils_llm_feature_fusion import crop
from experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_ids", nargs="+")
    parser.add_argument("-pl3d", "--planner3d", type=str, default="ExperimentPlanner3D_v21")
    parser.add_argument("-no_pp", action="store_true")
    parser.add_argument("-tl", type=int, required=False, default=8)
    parser.add_argument("-tf", type=int, required=False, default=8)
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true")
    parser.add_argument("-overwrite_plans", type=str, default=None, required=False)
    parser.add_argument("-overwrite_plans_identifier", type=str, default=None, required=False)

    args = parser.parse_args()
    dont_run_preprocessing = args.no_pp
    tl = args.tl
    tf = args.tf
    tasks = []

    for i in args.task_ids:
        task_name = convert_id_to_task_name(int(i))
        args.verify_dataset_integrity = False

        if args.verify_dataset_integrity:
            verify_dataset_integrity(join(nnUNet_raw_data, task_name))

        crop(task_name, False, tf)
        tasks.append(task_name)

    for t in tasks:
        print("\n\n\n", t)
        cropped_out_dir = os.path.join(nnUNet_cropped_data, t)
        preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)

        dataset_json = load_json(join(cropped_out_dir, "dataset.json"))
        modalities = list(dataset_json["modality"].values())
        collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
        dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=tf)
        _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)

        maybe_mkdir_p(preprocessing_output_dir_this_task)
        shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
        shutil.copy(join(nnUNet_raw_data, t, "dataset.json"), preprocessing_output_dir_this_task)

        threads = (tl, tf)
        print("number of threads: ", threads, "\n")

        exp_planner = ExperimentPlanner3D_v21(cropped_out_dir, preprocessing_output_dir_this_task)
        exp_planner.plan_experiment()

        if not dont_run_preprocessing:
            exp_planner.run_preprocessing(threads, "llm")


if __name__ == "__main__":
    main()
