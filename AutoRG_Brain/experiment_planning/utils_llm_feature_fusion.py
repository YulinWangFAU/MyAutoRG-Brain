from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p
from paths import nnUNet_raw_data, nnUNet_cropped_data
from preprocess.cropping_llm_feature_fusion import ImageCropper
import shutil
from configuration import default_num_threads
import json


def create_lists_from_splitted_dataset(base_folder_splitted):
    lists = []

    json_file = join(base_folder_splitted, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d["training"]

    for tr in training_files:
        cur_pat = list(tr.get("images", [tr["image"]]))
        cur_pat.append(tr["label1"])
        cur_pat.append(tr["label2"])
        lists.append(cur_pat)

    return lists, {int(i): d["modality"][str(i)] for i in d["modality"].keys()}


def crop(task_string, override=False, num_threads=default_num_threads):
    cropped_out_dir = join(nnUNet_cropped_data, task_string)
    maybe_mkdir_p(cropped_out_dir)

    if override and isdir(cropped_out_dir):
        shutil.rmtree(cropped_out_dir)
        maybe_mkdir_p(cropped_out_dir)

    splitted_4d_output_dir_task = join(nnUNet_raw_data, task_string)
    lists, _ = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

    imgcrop = ImageCropper(num_threads, cropped_out_dir)
    imgcrop.run_cropping(lists, overwrite_existing=override)
    shutil.copy(join(nnUNet_raw_data, task_string, "dataset.json"), cropped_out_dir)
