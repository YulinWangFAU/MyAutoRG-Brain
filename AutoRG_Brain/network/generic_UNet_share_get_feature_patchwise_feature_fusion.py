import numpy as np
import torch

from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regions

from network.generic_UNet_share_get_feature_patchwise import Generic_UNet as BaseGenericUNet


class Generic_UNet(BaseGenericUNet):
    # Input channels are built in script/build_feature_fusion_train_json.py as:
    #   0: t1n, 1: t1c, 2: t2w, 3: t2f
    # AutoRG has no separate T1CE encoder branch, so both t1n and t1c are routed
    # through the pretrained T1WI branch. The stacked feature order remains
    # [t1n-as-T1WI, t1c-as-T1WI, t2w-as-T2WI, t2f-as-T2FLAIR].
    input_channel_modalities = ("t1n", "t1c", "t2w", "t2f")
    modality_context_names = ("T1WI", "T1WI", "T2WI", "T2FLAIR")

    def _context_blocks_for_modal(self, modal):
        if modal == "DWI":
            return self.conv_blocks_context_a
        if modal == "T1WI":
            return self.conv_blocks_context_b
        if modal == "T2WI":
            return self.conv_blocks_context_c
        if modal == "T2FLAIR":
            return self.conv_blocks_context_d
        raise ValueError("Unsupported modal for feature fusion: %s" % modal)

    def _extract_region_features_single_modal(self, x, target, modal, region):
        if not x.shape[2] < x.shape[3] and not x.shape[2] < x.shape[4]:
            x = x.permute(0, 1, 4, 2, 3)
            target = target.permute(0, 1, 4, 2, 3)

        only_one_target = True if target.shape[1] == 1 else False
        skips = []
        conv_blocks_context = self._context_blocks_for_modal(modal)

        for d in range(len(conv_blocks_context) - 1):
            x = conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)

            if u == self.feature_layer:
                break

        for u in range(len(self.td) - 1 - self.feature_layer):
            target = self.td[u](target)

        region_features = []

        for b in range(len(x)):
            r = region[b]
            img_feature = x[b]

            b_target = target[b][:-1] if not only_one_target else target[b][-1:]
            a_target = target[b][-1:]

            ana_masks = []
            for ana_group in r:
                if ana_group == "global":
                    ana_masks.append(a_target)
                elif ana_group == "abnormal":
                    ana_masks.append(a_target)
                else:
                    ana_masks.append(np.zeros(b_target.shape))
                    for ana in ana_group:
                        ana_masks[-1] = np.logical_or(ana_masks[-1], b_target == ana)

            bboxes_ab = sk_regions(sk_label(a_target[0]))

            for idx, ana_group in enumerate(r):
                if ana_group == "global":
                    abnormal = np.ones(b_target.shape)
                elif ana_group == "abnormal":
                    abnormal = a_target
                else:
                    abnormal = np.zeros(b_target.shape)

                    for box in bboxes_ab:
                        z1, x1, y1, z2, x2, y2 = box.bbox

                        if len(np.where(ana_masks[idx][:, z1:z2 + 1, x1:x2 + 1, y1:y2 + 1])[0]):
                            abnormal[:, z1:z2 + 1, x1:x2 + 1, y1:y2 + 1] = 1

                    abnormal = np.logical_or(abnormal, ana_masks[idx])

                abnormal = torch.tensor(abnormal, dtype=torch.float16)
                abnormal = abnormal.repeat(img_feature.shape[0], 1, 1, 1).to(img_feature.device)
                abnormal_feature = abnormal * img_feature

                for pool_layer in self.pool_conv:
                    abnormal_feature = pool_layer(abnormal_feature)

                abnormal_feature = torch.flatten(abnormal_feature, start_dim=1, end_dim=3)
                abnormal_feature = abnormal_feature.permute(1, 0)

                region_features.append(abnormal_feature)

        return region_features, []

    def forward(self, x, target, modal, region, eval_mode_for_six="global", choose_dataset=None, **kwargs):
        if modal is None and x.shape[1] == 4:
            per_modal_features = []
            for modal_idx, modal_name in enumerate(self.modality_context_names):
                features, _ = self._extract_region_features_single_modal(
                    x[:, modal_idx:modal_idx + 1],
                    target,
                    modal_name,
                    region,
                )
                per_modal_features.append(features)

            fused_ready_features = []
            for region_idx in range(len(per_modal_features[0])):
                fused_ready_features.append(
                    torch.stack([features[region_idx] for features in per_modal_features], dim=0)
                )

            return fused_ready_features, []

        return self._extract_region_features_single_modal(x, target, modal, region)
