import torch
from torch import nn
from torch.optim import AdamW
from typing_extensions import override

from nnssl.architectures.build_architecture import build_network_architecture
from nnssl.architectures.gvsl_architecture import GVSLArchitecture
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.dataloading.data_loader_3d import nnsslCenterCropDataLoader3D
from nnssl.ssl_data.dataloading.gvsl_transform import GVSLTransform
from nnssl.training.loss.gvsl_loss import GVSLLoss

from nnssl.training.lr_scheduler.polylr import PolyLRScheduler
from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
from torch import autocast
from nnssl.utilities.helpers import dummy_context

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose

import matplotlib.pyplot as plt
import numpy as np

class GVSLTrainer(AbstractBaseTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)

        self.initial_lr = 1e-4
        self.num_iterations_per_epoch = 50

    @override
    def build_loss(self):
        return GVSLLoss()

    @override
    def build_architecture(
            self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        backbone = build_network_architecture(
            config_plan,
            num_input_channels,
            num_output_channels,
        )
        architecture = GVSLArchitecture(backbone, num_input_channels)

        return architecture

    @override
    def get_dataloaders(self):

        tr_transforms = self.get_training_transforms()
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_centercrop_dataloaders_with_doubled_batch_size()

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(
                self.num_iterations_per_epoch,
                data_loader=dl_tr,
                transform=tr_transforms,
                num_processes=allowed_num_processes,
                num_cached=6,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
            mt_gen_val = LimitedLenWrapper(
                self.num_val_iterations_per_epoch,
                data_loader=dl_val,
                transform=val_transforms,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=3,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
        return mt_gen_train, mt_gen_val


    def get_centercrop_dataloaders_with_doubled_batch_size(self):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnsslCenterCropDataLoader3D(
            dataset_tr,
            2*self.batch_size,
            self.config_plan.patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
        )
        dl_val = nnsslCenterCropDataLoader3D(
            dataset_val,
            2*self.batch_size,
            self.config_plan.patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
        )
        return dl_tr, dl_val

    @override
    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.network.parameters(),
            lr=self.initial_lr
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

        return optimizer, lr_scheduler

    @override
    def train_step(self, batch: dict) -> dict:
        imgsA = batch["imgsA"]
        imgsA_app = batch["imgsA_app"]
        imgsB = batch["imgsB"]

        # brain_image = imgsA_app[0][0]
        # depth_index = brain_image.shape[0] // 2
        # slice_2d = brain_image[depth_index, :, :]
        # slice_2d = slice_2d.numpy()
        # plt.figure(figsize=(6, 6))
        # plt.imshow(slice_2d, cmap='gray')
        # plt.title(f"2D Slice at Depth Index {depth_index}")
        # plt.axis('off')
        # plt.colorbar(label="Intensity")
        # plt.savefig("slice_visualization.png")
        # return {"loss": np.array(1)}

        imgsA = imgsA.to(self.device, non_blocking=True)
        imgsA_app = imgsA_app.to(self.device, non_blocking=True)
        imgsB = imgsB.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            recon_A, warped_BA, flow_BA = self.network(imgsA_app, imgsB)

        # NCC loss tends to get NANs with float16, thus we will not use autocast for loss calculation
        l = self.loss(imgsA, recon_A, warped_BA, flow_BA)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}


    @override
    def validation_step(self, batch: dict) -> dict:
        imgsA = batch["imgsA"]
        imgsA_app = batch["imgsA_app"]
        imgsB = batch["imgsB"]

        imgsA = imgsA.to(self.device, non_blocking=True)
        imgsA_app = imgsA_app.to(self.device, non_blocking=True)
        imgsB = imgsB.to(self.device, non_blocking=True)

        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                recon_A, warped_BA, flow_BA = self.network(imgsA_app, imgsB)

            # NCC loss tends to get NANs with float16, thus we will not use autocast for loss calculation
            l = self.loss(imgsA, recon_A, warped_BA, flow_BA)

        return {"loss": l.detach().cpu().numpy()}

    @staticmethod
    def get_training_transforms() -> AbstractTransform:
        tr_transforms = []

        tr_transforms.append(GVSLTransform(use_aug=True))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    @staticmethod
    def get_validation_transforms() -> AbstractTransform:
        return GVSLTrainer.get_training_transforms()


class GVSLTrainer_test(GVSLTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 2
        plan.configurations[configuration_name].patch_size = (96, 96, 96)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)


class GVSLTrainer_BS2(GVSLTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 2
        plan.configurations[configuration_name].patch_size = (180, 180, 180)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)


class GVSLTrainer_BS2_no_aug(GVSLTrainer_BS2):
    @staticmethod
    def get_training_transforms() -> AbstractTransform:
        tr_transforms = []

        tr_transforms.append(GVSLTransform(use_aug=False))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    @staticmethod
    def get_validation_transforms() -> AbstractTransform:
        return GVSLTrainer_BS2_no_aug.get_training_transforms()


