import yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import tqdm
import os
from einops import rearrange
from transformers import AutoModel
from batchgenerators.utilities.file_and_folder_operations import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score,confusion_matrix,average_precision_score

from time import time
from typing import Tuple, Union, List
import sys

if 'win' in sys.platform:
    #fix for windows platform
    import pathos
    Process = pathos.helpers.mp.Process
    Queue = pathos.helpers.mp.Queue
else:
    from multiprocessing import Process, Queue

from multiprocessing import Pool
from network_training.model_restore import load_model_and_checkpoint_files_llm
from network_training.model_restore import load_model_and_checkpoint_files
from utilities.llm_metric import *
from run.load_pretrained_weights import *
from dataset.utils import nnUNet_resize
from utilities.nd_softmax import *
import uuid
from inference.segmentation_export import save_segmentation_nifti_from_softmax, save_segmentation_nifti

import SimpleITK as sitk

from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regions

import re

class AutoRG_Brain():
    """
    Fused-feature inference version of AutoRG_Brain.

    Difference from the original inferenceSdk.py:
    - The original code generates one report immediately after each image/modality feature is extracted.
    - This fused version first collects image_hidden_states from multiple modalities belonging to the same case.
    - Then it stacks them into [B, M, P, D], for example [1, 4, 128, 1024].
    - The fused language model file network/language_model_patchwise_fused.py will perform mean fusion:
          [B, M, P, D] -> [B, P, D]
      before sending the fused features into the GPT2 decoder.

    Recommended usage:
    - Use this file together with network/language_model_patchwise_fused.py.
    - In the test script, import this class from inference.inferenceSdk_fused instead of inference.inferenceSdk.
    """

    def __init__(self, gpu_id, config):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id[0])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.trainer, params = load_model_and_checkpoint_files_llm(
            config['llm_folder'],
            mixed_precision=True,
            checkpoint_name=config['llm_chk']
        )

        self.segmodel = SegModel(config)

        self.trainer.load_checkpoint_ram(params[0], False)
        load_pretrained_weights(
            self.trainer.network,
            join(config['seg_folder'], config['seg_chk'] + '.model')
        )

        self.num_threads_preprocessing = 6
        self.step_size = 0.5
        self.mixed_precision = True
        self.num_threads_nifti_save = 2

        self.hammer_anas = json.load(open('utils_file/hammer_anas.json', 'r'))
        self.eval_mode = config['eval_mode']

        # Fused inference settings.
        # If your test JSON contains multiple cases, use a shared case_id/patient_id/subject_id/study_id
        # for the four modalities of the same case. If no such key exists, this code tries to infer
        # the case id from the parent folder or image filename.
        self.fused_modal_order = config.get(
            'fused_modal_order',
            ['T1WI', 'T1W', 'T1N', 'T1CE', 'T1C', 'T2WI', 'T2W', 'T2', 'FLAIR', 'T2F']
        )
        self.fused_debug = bool(config.get('debug_fused', False))

    # =========================================================
    # Helper functions for fused inference
    # =========================================================

    def _strip_nii_suffix(self, filename):
        """Remove .nii.gz or normal suffixes from a file name."""
        name = os.path.basename(filename)
        if name.endswith('.nii.gz'):
            return name[:-7]
        return os.path.splitext(name)[0]

    def _infer_case_id(self, item):
        """
        Infer which images belong to the same patient/case.

        Priority:
        1. Explicit keys in the JSON: case_id / patient_id / subject_id / study_id
        2. Parent folder name, which is usually the BraTS case id
        3. Filename after removing common modality suffixes such as -t1n, -t1c, -t2w, -t2f
        """
        for key in ['case_id', 'patient_id', 'subject_id', 'study_id']:
            if key in item and item[key] is not None and str(item[key]).strip() != '':
                return str(item[key])

        image_path = item['image']
        parent_name = os.path.basename(os.path.dirname(image_path))
        stem = self._strip_nii_suffix(image_path)

        # BraTS-style files usually look like:
        #   BraTS-GLI-00006-000-t1n.nii.gz
        # and are stored in a parent folder:
        #   BraTS-GLI-00006-000/
        # In this situation, the parent folder is the cleanest case id.
        if parent_name and parent_name not in ['.', '/', '']:
            modality_suffix_pattern = r'(?i)([-_](t1wi|t1w|t1n|t1ce|t1c|t2wi|t2w|t2|t2f|flair|seg|mask))$'
            if re.search(modality_suffix_pattern, stem):
                return parent_name

        # Fallback: remove common modality suffixes from the filename.
        case_id = re.sub(
            r'(?i)([-_](t1wi|t1w|t1n|t1ce|t1c|t2wi|t2w|t2|t2f|flair|seg|mask))$',
            '',
            stem
        )
        return case_id

    def _modal_rank(self, modal):
        """Return a stable sorting rank for modalities."""
        modal_upper = str(modal).upper()
        for idx, name in enumerate(self.fused_modal_order):
            if modal_upper == str(name).upper():
                return idx
        return len(self.fused_modal_order)

    def _to_batched_patch_feature(self, region_features):
        """
        Convert region_features to a tensor shape accepted by the fused language model.

        Expected target shape before stacking modalities:
            [B, P, D], usually [1, 128, 1024]

        The original pipeline often returns a list of tensors. We first convert it to a tensor.
        Depending on the upstream code, the resulting tensor may be [P, D] or [B, P, D].
        """
        region_features = torch.tensor(
            np.array([item.cpu().detach().numpy() for item in region_features]),
            dtype=torch.float32
        ).to(self.trainer.llm_model.device)

        if region_features.dim() == 2:
            # [P, D] -> [1, P, D]
            region_features = region_features.unsqueeze(0)
        elif region_features.dim() == 3:
            # already [B, P, D]
            pass
        else:
            raise ValueError(
                f"region_features should be [P, D] or [B, P, D], "
                f"but got shape {tuple(region_features.shape)}"
            )

        return region_features

    def _generate_report_from_features(self, fused_or_multimodal_features):
        """
        Generate a report from either:
        - [B, P, D], already fused/single-modal features
        - [B, M, P, D], multi-modal features before fusion

        The actual [B, M, P, D] -> [B, P, D] mean fusion is handled inside
        network/language_model_patchwise_fused.py.
        """
        print("\n[FUSED INFERENCE] features entering llm_model.generate:", tuple(fused_or_multimodal_features.shape))
        print("[FUSED INFERENCE] device:", fused_or_multimodal_features.device)

        output = self.trainer.llm_model.generate(
            fused_or_multimodal_features,
            max_length=300,
            num_beams=1,
            num_beam_groups=1,
            do_sample=False,
            num_return_sequences=1,
            early_stopping=True
        )

        generated_sents = self.trainer.tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return " ".join([s.strip() for s in generated_sents if s is not None and s.strip() != ""])

    def report(self, input_case_dict):
        """
        Fused report generation.

        The input JSON can still contain one item per image/modality, for example:

        [
          {"image": "...-t1n.nii.gz", "modal": "T1WI", "label": "...seg.nii.gz"},
          {"image": "...-t1c.nii.gz", "modal": "T1WI", "label": "...seg.nii.gz"},
          {"image": "...-t2w.nii.gz", "modal": "T2WI", "label": "...seg.nii.gz"},
          {"image": "...-t2f.nii.gz", "modal": "FLAIR", "label": "...seg.nii.gz"}
        ]

        This function will collect all features from the same inferred case_id and generate
        one global report per case.
        """
        test_file = input_case_dict

        list_of_lists = [[j['image']] for j in test_file]
        list_of_ab_segs = [j['label'] if 'label' in j else None for j in test_file]
        list_of_ana_segs = [j['label2'] if 'label2' in j else None for j in test_file]
        list_of_reports = None if 'report' not in test_file[0] else [j['report'] for j in test_file]
        modals = [j['modal'] for j in test_file]

        # Build metadata for grouping after preprocessing.
        item_metadata = []
        case_identifiers = []
        for idx, item in enumerate(test_file):
            image_stem = self._strip_nii_suffix(item['image'])
            case_id = self._infer_case_id(item)
            identifier = f"{idx:04d}__{image_stem}"
            case_identifiers.append(identifier)
            item_metadata.append({
                'identifier': identifier,
                'case_id': case_id,
                'image': item['image'],
                'modal': item['modal'],
                'input_index': idx,
            })

        metadata_by_identifier = {m['identifier']: m for m in item_metadata}
        case_order = []
        for m in item_metadata:
            if m['case_id'] not in case_order:
                case_order.append(m['case_id'])

        print("\n[FUSED INFERENCE] Number of input images/modalities:", len(test_file))
        print("[FUSED INFERENCE] Number of inferred cases:", len(case_order))
        for cid in case_order:
            cur = [m for m in item_metadata if m['case_id'] == cid]
            print(f"[FUSED INFERENCE] case_id={cid}: {len(cur)} modality item(s)")
            for m in cur:
                print(f"    - index={m['input_index']}, modal={m['modal']}, image={m['image']}")

        # Run segmentation first, same as the original pipeline.
        list_of_ab_segs, list_of_ana_segs = self.segmodel.seg(
            list_of_lists,
            list_of_ab_segs,
            list_of_ana_segs,
            modals
        )

        print("emptying cuda cache")
        torch.cuda.empty_cache()

        preprocessing = preprocess_multithreaded(
            self.trainer,
            list_of_lists,
            list_of_ab_segs,
            list_of_ana_segs,
            list_of_reports,
            case_identifiers,
            modals,
            self.num_threads_preprocessing
        )

        setup_seed(42)
        self.trainer.network.eval()
        self.trainer.llm_model.eval()

        # Collect features by case id instead of generating immediately.
        features_by_case = {}
        meta_by_case = {}

        for preprocessed in preprocessing:
            identifier, modal, the_image_path, the_ab_seg_path, the_ana_seg_path, (r, d, s_ab, s_ana, dct) = preprocessed
            meta = metadata_by_identifier[identifier]
            case_id = meta['case_id']

            d = np.expand_dims(nnUNet_resize(d[0], self.trainer.patch_size, axis=0), axis=0)

            s_ab = nnUNet_resize(
                s_ab[0],
                self.trainer.patch_size,
                is_seg=True,
                axis=0
            ) if s_ab is not None else np.zeros(self.trainer.patch_size)
            s_ab = np.expand_dims(s_ab, axis=0)

            s_ana = nnUNet_resize(
                s_ana[0],
                self.trainer.patch_size,
                is_seg=True,
                axis=0
            ) if s_ana is not None else np.zeros(self.trainer.patch_size)
            s_ana = np.expand_dims(s_ana, axis=0)

            s = np.concatenate((s_ana, s_ab), axis=0)

            if r is not None:
                regions, gt_reports = self.trainer.split_batch_report([r])
            else:
                regions, gt_reports = None, None

            region_features, region_direction_names = self.trainer.predict_preprocessed_data_return_region_report(
                d,
                s,
                regions,
                do_mirroring=False,
                mirror_axes=self.trainer.data_aug_params['mirror_axes'],
                use_sliding_window=True,
                step_size=self.step_size,
                use_gaussian=True,
                all_in_gpu=False,
                mixed_precision=self.mixed_precision,
                modal=modal,
                eval_mode=self.eval_mode
            )

            region_features = self._to_batched_patch_feature(region_features)

            print("\n[FUSED INFERENCE] Extracted feature")
            print("  case_id:", case_id)
            print("  modal:", modal)
            print("  image:", the_image_path)
            print("  region_features shape:", tuple(region_features.shape))
            print("  region_features device:", region_features.device)

            features_by_case.setdefault(case_id, []).append({
                'feature': region_features.detach(),
                'modal': modal,
                'input_index': meta['input_index'],
            })

            meta_by_case.setdefault(case_id, {
                'images': [],
                'modals': [],
                'ab_masks': [],
                'ana_masks': [],
            })
            meta_by_case[case_id]['images'].append(the_image_path)
            meta_by_case[case_id]['modals'].append(modal)
            meta_by_case[case_id]['ab_masks'].append(the_ab_seg_path)
            meta_by_case[case_id]['ana_masks'].append(the_ana_seg_path)

            # Free local arrays as early as possible.
            del d, s, s_ab, s_ana, region_features
            torch.cuda.empty_cache()

        # Generate one global report per case.
        pred_report = []

        for case_id in case_order:
            if case_id not in features_by_case:
                print(f"[FUSED INFERENCE WARNING] No features found for case_id={case_id}. Skipping.")
                continue

            feature_items = features_by_case[case_id]

            # Stable order: first modality rank, then original input order.
            feature_items = sorted(
                feature_items,
                key=lambda x: (self._modal_rank(x['modal']), x['input_index'])
            )

            feature_list = [x['feature'] for x in feature_items]
            modal_list = [x['modal'] for x in feature_items]

            print("\n[FUSED INFERENCE] Preparing fused generation for case:", case_id)
            print("[FUSED INFERENCE] modality order:", modal_list)
            for idx, feat in enumerate(feature_list):
                print(f"  feature[{idx}] shape:", tuple(feat.shape))

            if len(feature_list) == 1:
                # Single modality fallback: [B, P, D]
                multimodal_or_single_feature = feature_list[0]
                print("[FUSED INFERENCE] Only one modality found. Using single feature directly.")
            else:
                # Multi-modal feature stack: list of [B, P, D] -> [B, M, P, D]
                try:
                    multimodal_or_single_feature = torch.stack(feature_list, dim=1)
                except RuntimeError as e:
                    shapes = [tuple(f.shape) for f in feature_list]
                    raise RuntimeError(
                        f"Cannot stack modality features for case_id={case_id}. "
                        f"All modality features must have the same shape. Current shapes: {shapes}"
                    ) from e

                print("[FUSED INFERENCE] stacked multimodal feature shape:", tuple(multimodal_or_single_feature.shape))
                print("[FUSED INFERENCE] This should be [B, M, P, D], e.g. [1, 4, 128, 1024].")

            pred_global_report = self._generate_report_from_features(multimodal_or_single_feature)

            case_meta = meta_by_case[case_id]
            pred_report.append({
                'case_id': case_id,
                'images': case_meta['images'],
                'modals': case_meta['modals'],
                'pred_report': pred_global_report,
                'ab_masks': case_meta['ab_masks'],
                'ana_masks': case_meta['ana_masks'],
                'fusion': 'mean_feature_fusion_inside_language_model',
                'feature_shape_before_generate': tuple(multimodal_or_single_feature.shape),
            })

            del multimodal_or_single_feature
            torch.cuda.empty_cache()

        print("inference done.")
        return pred_report

class SegModel():
    def __init__(self, config):
        self.trainer, params = load_model_and_checkpoint_files(config['seg_folder'], mixed_precision=True,
                                                        checkpoint_name=config['seg_chk'])
        self.trainer.load_checkpoint_ram(params[0], False)
        self.num_threads_nifti_save = 2
        self.num_threads_preprocessing = 6
        self.output_mask_dir = config['output_dir']

    def seg(self, list_of_lists, list_of_ab_segs, list_of_ana_segs, modals):
        pool = Pool(self.num_threads_nifti_save)
        results = []

        output_ab_filenames = []
        output_ana_filenames = []
        ab_flag = []
        ana_flag = []
        for idx,item in enumerate(list_of_lists):
            # uid = uuid.uuid1().hex
            img_path = item[0].split('/')[-1].split('.')[0]
            if list_of_ab_segs[idx] is not None:
                ab_flag.append(True)
                output_ab_filenames.append(list_of_ab_segs[idx])
            else:
                ab_flag.append(False)
                output_ab_filenames.append(join(self.output_mask_dir,img_path+'_ab.nii.gz'))
            if list_of_ana_segs[idx] is not None:
                ana_flag.append(True)
                output_ana_filenames.append(list_of_ana_segs[idx])
            else:
                ana_flag.append(False)
                output_ana_filenames.append(join(self.output_mask_dir,img_path+'_ana.nii.gz'))
        
        print("emptying cuda cache")
        torch.cuda.empty_cache()

        if 'segmentation_export_params' in self.trainer.plans.keys():
            force_separate_z = self.trainer.plans['segmentation_export_params']['force_separate_z']
            interpolation_order = self.trainer.plans['segmentation_export_params']['interpolation_order']
            interpolation_order_z = self.trainer.plans['segmentation_export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0
        
        print("starting preprocessing generator")
        # under 3dfullres setting, seg_from_prev_stage is None
        preprocessing = preprocess_multithreaded_seg(self.trainer, list_of_lists, modals, output_ab_filenames, output_ana_filenames, ab_flag, ana_flag, self.num_threads_preprocessing)
        print("starting prediction...")

        result_modal = []

        for preprocessed in preprocessing:

            output_ab_filename, output_ana_filename, is_exist_ab, is_exist_ana, image_path, modal, (d, s, dct) = preprocessed

            if is_exist_ab and is_exist_ana:
                continue

            print("predicting", output_ab_filename, output_ana_filename, "modal",modal)
            
            if isinstance(d, str):
                data = np.load(d)
                os.remove(d)
                d = data

            # load the params of the network

            softmaxs = self.trainer.predict_preprocessed_data_return_seg_and_softmax(
                d, do_mirroring=False, mirror_axes=self.trainer.data_aug_params['mirror_axes'], use_sliding_window=True,
                step_size=0.5, use_gaussian=True, all_in_gpu=False,
                mixed_precision=True, modal=modal)
            
            softmax_abnormal, softmax_anatomy = softmaxs[1], softmaxs[3]

            transpose_forward = self.trainer.plans.get('transpose_forward')
            if transpose_forward is not None:
                transpose_backward = self.trainer.plans.get('transpose_backward')
                softmax_abnormal = softmax_abnormal.transpose([0] + [i + 1 for i in transpose_backward]) # 2, x ,y z
                softmax_anatomy = softmax_anatomy.transpose([0] + [i + 1 for i in transpose_backward]) # 96, x, y, z
            
            # if list_of_segs is not None:
            #     gt = s[0]
            #     gt[gt<0] = 0

            #     pred = softmax_abnormal.argmax(0)
            #     gt[gt>0] = 1
            #     gt[gt<0] = 0
            
            npz_file = None

            if hasattr(self.trainer, 'regions_class_order'):
                region_class_order = self.trainer.regions_class_order
            else:
                region_class_order = None

            if not is_exist_ab:
                results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                ((softmax_abnormal, output_ab_filename, dct, interpolation_order, region_class_order,
                                                    None, None,
                                                    npz_file, None, force_separate_z, interpolation_order_z),)
                                                    ))
            if not is_exist_ana:
                results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                ((softmax_anatomy, output_ana_filename, dct, interpolation_order, region_class_order,
                                                    None, None,
                                                    npz_file, None, force_separate_z, interpolation_order_z, True),)
                                                ))

        pool.close()
        pool.join()

        return output_ab_filenames, output_ana_filenames

def _get_bert_basemodel(bert_model_name):
    try:
        model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
        print("text feature extractor:", bert_model_name)
    except:
        raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

    for param in model.parameters():
        param.requires_grad = False

    return model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def preprocess_save_to_queue_seg(preprocess_fn, q, list_of_lists, modals, output_ab_files, output_ana_files, ab_flags, ana_flags, transpose_forward):
    # suppress output
    # sys.stdout = open(os.devnull, 'w')

    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            output_ab_file = output_ab_files[i]
            output_ana_file = output_ana_files[i]
            ab_flag = ab_flags[i]
            ana_flag = ana_flags[i]
            print("preprocessing", l)
            d, s, dct = preprocess_fn(l, None, target_shape=None)

            if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                np.save(output_file[:-7] + ".npy", d)
                d = output_file[:-7] + ".npy"
            modal = modals[i]
            q.put((output_ab_file, output_ana_file, ab_flag, ana_flag, l, modal, (d, s, dct)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print("error in", l)
            print(e)
    q.put("end")
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    else:
        print("This worker has ended successfully, no errors to report")


def preprocess_multithreaded_seg(trainer, list_of_lists, modals, output_ab_files, output_ana_files, ab_flags, ana_flags, num_processes=2):

    # num_processes default = 6
    num_processes = min(len(list_of_lists), num_processes)

    # classes = list(range(1, trainer.num_classes)) # 96

    # assert isinstance(trainer, nnUNetTrainer)
    q = Queue(1)
    processes = []

    for i in range(num_processes):

        pr = Process(target=preprocess_save_to_queue_seg, args=(trainer.preprocess_patient, q,
                                                            list_of_lists[i::num_processes],
                                                            modals[i::num_processes],
                                                            output_ab_files[i::num_processes], output_ana_files[i::num_processes], ab_flags[i::num_processes], ana_flags[i::num_processes], trainer.plans['transpose_forward']))
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == "end":
                end_ctr += 1
                continue
            else:
                yield item

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join()

        q.close()
    
def preprocess_save_to_queue(preprocess_fn, q, list_of_lists, list_of_ab_segs, list_of_ana_segs, list_of_reports, case_identifiers, modals, transpose_forward):

    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            the_ab_seg = list_of_ab_segs[i] if list_of_ab_segs is not None else None
            the_ana_seg = list_of_ana_segs[i] if list_of_ana_segs is not None else None

            target_shape = None

            if the_ab_seg is not None:
                d, s_ab, dct = preprocess_fn(l, the_ab_seg, target_shape=target_shape)
            else:
                s_ab = None
            
            if the_ana_seg is not None:
                d, s_ana, dct = preprocess_fn(l, the_ana_seg, target_shape=target_shape)
            else:
                s_ana = None

            if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                # np.save(output_file[:-7] + ".npy", d)
                # d = output_file[:-7] + ".npy"
                print(l)
            r = list_of_reports[i] if list_of_reports is not None else None
            identi = case_identifiers[i]
            modal = modals[i]
            q.put((identi, modal, l, the_ab_seg, the_ana_seg, (r, d, s_ab, s_ana, dct)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print("error in", l)
            print(e)
    q.put("end")
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    else:
        print("This worker has ended successfully, no errors to report")


def preprocess_multithreaded(trainer, list_of_lists, list_of_ab_segs, list_of_ana_segs, list_of_reports, case_identifiers, modals, num_processes=6):

    # num_processes default = 6
    num_processes = min(len(list_of_lists), num_processes)

    q = Queue(1)
    processes = []

    for i in range(num_processes):
        the_ab_segs = list_of_ab_segs[i::num_processes] if list_of_ab_segs is not None else None
        the_ana_segs = list_of_ana_segs[i::num_processes] if list_of_ana_segs is not None else None
        the_reports = list_of_reports[i::num_processes] if list_of_reports is not None else None
        pr = Process(target=preprocess_save_to_queue, args=(trainer.preprocess_patient, q,
                                                            list_of_lists[i::num_processes], the_ab_segs, the_ana_segs, the_reports,case_identifiers[i::num_processes],modals[i::num_processes],
                                                            trainer.plans['transpose_forward']))
        # pr = Process(target=preprocess_save_to_queue, args=(trainer.preprocess_patient, q,
        #                                                     list_of_lists, list_of_ab_segs, list_of_ana_segs, list_of_reports,case_identifiers,modals,
        #                                                     trainer.plans['transpose_forward']))
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == "end":
                end_ctr += 1
                continue
            else:
                yield item

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join()

        q.close()
