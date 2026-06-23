import os

import dataset.dataset_loading_llm_multi as multi_loader

if not hasattr(multi_loader, "DataLoader3D_Multi"):
    multi_loader.DataLoader3D_Multi = multi_loader.DataLoader3D

from network_training.nnUNetTrainerV2_llm_resize_multi import nnUNetTrainerV2 as BaseTrainer
from network_training.nnUNetTrainerV2_llm_resize_multi import (
    DataLoader3D,
    GPT2Tokenizer,
    InitWeights_He,
    LanguageModel_patch,
    load_pickle,
    nn,
    np,
    softmax_helper,
    torch,
)
from network.generic_UNet_share_get_feature_patchwise_feature_fusion import Generic_UNet as Generic_UNet_share_patch


class nnUNetTrainerV2(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = int(os.environ.get("AUTORG_REAL_BATCH_SIZE", "2"))
        print("[FEATURE FUSION] real dataloader batch_size:", self.batch_size)

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        dl_tr = DataLoader3D(
            self.dataset_tr,
            self.basic_generator_patch_size,
            self.patch_size,
            self.batch_size,
            report=self.report["training"],
            has_prev_stage=True,
            oversample_foreground_percent=self.oversample_foreground_percent,
            pad_mode="constant",
            pad_sides=self.pad_all_sides,
            memmap_mode="r",
            input_mode=self.input_mode,
            single_modal_index=self.single_modal_index,
        )

        dl_val = DataLoader3D(
            self.dataset_val,
            self.patch_size,
            self.patch_size,
            self.batch_size,
            report=self.report["validation"],
            has_prev_stage=True,
            oversample_foreground_percent=self.oversample_foreground_percent,
            pad_mode="constant",
            pad_sides=self.pad_all_sides,
            memmap_mode="r",
            input_mode=self.input_mode,
            single_modal_index=self.single_modal_index,
        )
        return dl_tr, dl_val

    def get_tokenizer(self):
        checkpoint = "healx/gpt-2-pubmed-medium"
        tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def initialize_network(self):
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {"eps": 1e-5, "affine": True}
        dropout_op_kwargs = {"p": 0, "inplace": True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}

        if self.network_type == "normal":
            raise ValueError("feature_fusion trainer expects --network_type share")

        plans = load_pickle("utils_file/nnUNetPlansv2.1_plans_3D.pkl")
        num_input_channels = plans["num_modalities"]
        base_num_features = plans["base_num_features"]
        net_num_pool_op_kernel_sizes = plans["plans_per_stage"][0]["pool_op_kernel_sizes"]
        conv_per_stage = plans["conv_per_stage"]
        net_conv_kernel_sizes = plans["plans_per_stage"][0]["conv_kernel_sizes"]

        self.network = Generic_UNet_share_patch(
            num_input_channels, base_num_features, 96, 2,
            len(net_num_pool_op_kernel_sizes),
            conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
            dropout_op_kwargs,
            net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
            net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True,
            feature_layer=self.feature_layer, size=self.size
        )

        if not self.train_with_seg:
            for name, parameter in self.network.named_parameters():
                if name.startswith("pool_conv"):
                    parameter.requires_grad = True
                else:
                    parameter.requires_grad = False

        self.llm_model = LanguageModel_patch(self.network.img_patch_num, self.max_tokens)
        self.tokenizer = self.get_tokenizer()

        if torch.cuda.is_available():
            self.network.cuda()
            self.llm_model.cuda()
        self.network.inference_apply_nonlin = softmax_helper
