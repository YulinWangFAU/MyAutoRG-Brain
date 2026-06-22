from preprocess.cropping_llm import *


MODALITY_SUFFIXES = ("t1n", "t1c", "t2w", "t2f")


def get_case_identifier(case):
    stem = os.path.basename(case[0]).split(".nii.gz")[0]
    for suffix in MODALITY_SUFFIXES:
        marker = "-" + suffix
        if stem.endswith(marker):
            return stem[:-len(marker)]
    return stem


class ImageCropper(ImageCropper):
    def load_crop_save(self, case, case_identifier, overwrite_existing=False):
        try:
            print(case_identifier)
            if overwrite_existing \
                    or (not os.path.isfile(os.path.join(self.output_folder, "%s.npz" % case_identifier))
                        or not os.path.isfile(os.path.join(self.output_folder, "%s.pkl" % case_identifier))):

                data_files = case[:-2]
                data, seg, properties = self.crop_from_list_of_files(data_files, case[-2], case[-1])
                all_data = np.vstack((data, seg))
                np.savez_compressed(os.path.join(self.output_folder, "%s.npz" % case_identifier), data=all_data)
                with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'wb') as f:
                    pickle.dump(properties, f)
        except Exception as e:
            print("Exception in", case_identifier, ":")
            print(e)
            raise e

    def run_cropping(self, list_of_files, overwrite_existing=False, output_folder=None):
        if output_folder is not None:
            self.output_folder = output_folder

        output_folder_gt = os.path.join(self.output_folder, "gt_segmentations")
        maybe_mkdir_p(output_folder_gt)
        for case in list_of_files:
            if case[-2] is not None:
                shutil.copy(case[-2], output_folder_gt)

        list_of_args = []
        for case in list_of_files:
            case_identifier = get_case_identifier(case)
            list_of_args.append((case, case_identifier, overwrite_existing))

        p = Pool(self.num_threads)
        p.starmap(self.load_crop_save, list_of_args)
        p.close()
        p.join()
