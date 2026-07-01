import torch
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from torch import nn
from torch.optim import AdamW
from typing_extensions import override

from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.gvsl_architecture import GVSLArchitecture, GVSLArchitecture_recon_only
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.dataloading.data_loader_3d import nnsslCenterCropDataLoader3D
from nnssl.ssl_data.dataloading.gvsl_transform import GVSLTransform, SpatialTransforms
from nnssl.training.loss.gvsl_loss import GVSLLoss, L_mse

from nnssl.training.lr_scheduler.polylr import PolyLRScheduler
from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
from torch import autocast
from nnssl.utilities.helpers import dummy_context

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from typing import List

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
        do_spatial_aug: bool = True,
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)

        self.do_spatial_aug = do_spatial_aug
        self.spatial_transforms = SpatialTransforms()

    @override
    def build_loss(self):
        return GVSLLoss()

    @override
    def build_architecture_and_adaptation_plan(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        architecture = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
        )
        architecture = GVSLArchitecture(architecture, num_input_channels)
        raise NotImplementedError("Missing adaptation plan")
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
            2 * self.batch_size,
            self.config_plan.patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
        )
        dl_val = nnsslCenterCropDataLoader3D(
            dataset_val,
            2 * self.batch_size,
            self.config_plan.patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
        )
        return dl_tr, dl_val

    def visualize_brain_slices(self, batch_tensor, save_path, row_view=False):
        """
        Visualizes and saves 2D slices of 3D brain images from a batch tensor.

        Parameters:
        - batch_tensor (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).
        - save_path (str): Path to save the visualization.
        - row_view (bool): If True, arrange slices in a row; otherwise, arrange them in a column.
        """
        assert batch_tensor.dim() == 5, "Expected input tensor shape: (batch_size, channels, depth, height, width)"
        batch_size = batch_tensor.size(0)

        slices = []
        for i in range(batch_size):
            brain_image = batch_tensor[i][0]  # Assume first channel is the relevant one
            depth_index = brain_image.shape[0] // 2  # Middle depth index
            slice_2d = brain_image[depth_index, :, :].cpu().numpy()  # Convert to numpy for plotting
            slices.append(slice_2d)

        # Determine figure layout
        if row_view:
            nrows, ncols = 1, batch_size
        else:
            nrows, ncols = batch_size, 1

        # Plot slices
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))
        axes = np.atleast_1d(axes)  # Ensure axes is always iterable

        for i, (ax, slice_2d) in enumerate(zip(axes.flatten(), slices)):
            ax.imshow(slice_2d, cmap="gray")
            ax.set_title(f"Sample {i + 1} (Depth {depth_index})")
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.ncc_losses = []
        self.mse_losses = []
        self.smooth_losses = []

    def on_train_epoch_end(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)
        self.print_to_log_file(
            f"NCC_loss: {np.mean(self.ncc_losses)} | "
            f"MSE_loss: {np.mean(self.mse_losses)} | "
            f"SMOOTH_loss {np.mean(self.smooth_losses)}"
        )

    @override
    def train_step(self, batch: dict) -> dict:
        imgsA = batch["imgsA"]
        imgsA_app = batch["imgsA_app"]
        imgsB = batch["imgsB"]

        imgsA = imgsA.to(self.device, non_blocking=True)
        imgsA_app = imgsA_app.to(self.device, non_blocking=True)
        imgsB = imgsB.to(self.device, non_blocking=True)

        with torch.device(self.device):
            # For some reason, the official implementation includes affine transformations and deformations as
            # data augmentations. Mentioned nowhere in the paper...
            # These augmentations benefit from GPU acceleration, and since batchgenerators does not provide GPU support
            # for their transforms, they have to be conducted here
            if self.do_spatial_aug:
                affine_mat, flow = self.spatial_transforms.get_rand_spatial(
                    self.batch_size, self.config_plan.patch_size
                )
                imgsA = self.spatial_transforms.augment_spatial(imgsA, affine_mat, flow)
                imgsA_app = self.spatial_transforms.augment_spatial(imgsA_app, affine_mat, flow)
                imgsB = self.spatial_transforms.augment_spatial(imgsB, affine_mat, flow)

            # self.visualize_brain_slices(imgsA, "imgsA.png")
            # self.visualize_brain_slices(imgsA_app, "imgsA_app.png")
            # self.visualize_brain_slices(imgsB, "imgsB.png")
            # exit()
            # return {"loss": np.array(1)}

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                recon_A, warped_BA, flow_BA = self.network(imgsA_app, imgsB)

            # NCC loss tends to get NANs with float16, thus we will not use autocast for loss calculation
            # l = self.loss(imgsA, recon_A, warped_BA, flow_BA)
            ncc, mse, smooth = self.loss(imgsA, recon_A, warped_BA, flow_BA)
            self.ncc_losses.append(ncc.detach().item())
            self.mse_losses.append(mse.detach().item())
            self.smooth_losses.append(smooth.detach().item())
            l = ncc + mse + smooth

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

        with torch.no_grad(), torch.device(self.device):
            if self.do_spatial_aug:
                affine_mat, flow = self.spatial_transforms.get_rand_spatial(
                    self.batch_size, self.config_plan.patch_size
                )
                imgsA = self.spatial_transforms.augment_spatial(imgsA, affine_mat, flow)
                imgsA_app = self.spatial_transforms.augment_spatial(imgsA_app, affine_mat, flow)
                imgsB = self.spatial_transforms.augment_spatial(imgsB, affine_mat, flow)

            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                recon_A, warped_BA, flow_BA = self.network(imgsA_app, imgsB)

            # l = self.loss(imgsA, recon_A, warped_BA, flow_BA)
            ncc, mse, smooth = self.loss(imgsA, recon_A, warped_BA, flow_BA)
            l = ncc + mse + smooth

        return {"loss": l.detach().cpu().numpy()}

    @staticmethod
    def get_training_transforms() -> AbstractTransform:
        tr_transforms = []

        tr_transforms.append(GVSLTransform())
        tr_transforms.append(NumpyToTensor(cast_to="float", keys=["imgsA", "imgsA_app", "imgsB"]))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    @staticmethod
    def get_validation_transforms() -> AbstractTransform:
        return GVSLTrainer.get_training_transforms()


class GVSLTrainer_do_spatial(GVSLTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device, do_spatial_aug=True)


class GVSLTrainer_no_spatial(GVSLTrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device, do_spatial_aug=False)


class GVSLTrainer_test(GVSLTrainer_do_spatial):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (192, 192, 192)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 4
        self.num_iterations_per_epoch = 20
        self.num_val_iterations_per_epoch = 2


class GVSLTrainer_BS2_lr_1e4(GVSLTrainer_do_spatial):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 2
        self.initial_lr = 1e-4


class GVSLTrainer_BS2_lr_1e5(GVSLTrainer_do_spatial):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 2
        self.initial_lr = 1e-5


class GVSLTrainer_recon_only(GVSLTrainer_no_spatial):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 2
        self.initial_lr = 1e-4
        self.num_iterations_per_epoch = 100

    def build_architecture_and_adaptation_plan(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        backbone = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
            encoder_only=True,
        )
        architecture = GVSLArchitecture_recon_only(backbone, num_input_channels)
        raise NotImplementedError("Missing adaptation plan")
        return architecture

    @override
    def train_step(self, batch: dict) -> dict:
        imgsA = batch["imgsA"]
        imgsA_app = batch["imgsA_app"]
        imgsB = batch["imgsB"]

        imgsA = imgsA.to(self.device, non_blocking=True)
        imgsA_app = imgsA_app.to(self.device, non_blocking=True)
        imgsB = imgsB.to(self.device, non_blocking=True)

        with torch.device(self.device):
            # For some reason, the official implementation includes affine transformations and deformations as
            # data augmentations. Mentioned nowhere in the paper...
            # These augmentations benefit from GPU acceleration, and since batchgenerators does not provide GPU support
            # for their transforms, they have to be conducted here
            if self.do_spatial_aug:
                affine_mat, flow = self.spatial_transforms.get_rand_spatial(
                    self.batch_size, self.config_plan.patch_size
                )
                imgsA = self.spatial_transforms.augment_spatial(imgsA, affine_mat, flow)
                imgsA_app = self.spatial_transforms.augment_spatial(imgsA_app, affine_mat, flow)
                imgsB = self.spatial_transforms.augment_spatial(imgsB, affine_mat, flow)

            # self.visualize_brain_slices(imgsA, "imgsA_no_aug.png")
            # self.visualize_brain_slices(imgsB, "imgsB_no_aug.png")
            # return {"loss": np.array(1)}

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                recon_A = self.network(imgsA_app, imgsB)

            # NCC loss tends to get NANs with float16, thus we will not use autocast for loss calculation
            # l = self.loss(imgsA, recon_A, warped_BA, flow_BA)
            mse = L_mse(imgsA, recon_A)
            self.ncc_losses.append(0)
            self.mse_losses.append(mse.detach().item())
            self.smooth_losses.append(0)
            l = mse

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

    def validation_step(self, batch: dict) -> dict:
        imgsA = batch["imgsA"]
        imgsA_app = batch["imgsA_app"]
        imgsB = batch["imgsB"]

        imgsA = imgsA.to(self.device, non_blocking=True)
        imgsA_app = imgsA_app.to(self.device, non_blocking=True)
        imgsB = imgsB.to(self.device, non_blocking=True)

        with torch.no_grad(), torch.device(self.device):
            if self.do_spatial_aug:
                affine_mat, flow = self.spatial_transforms.get_rand_spatial(
                    self.batch_size, self.config_plan.patch_size
                )
                imgsA = self.spatial_transforms.augment_spatial(imgsA, affine_mat, flow)
                imgsA_app = self.spatial_transforms.augment_spatial(imgsA_app, affine_mat, flow)
                imgsB = self.spatial_transforms.augment_spatial(imgsB, affine_mat, flow)

            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                recon_A = self.network(imgsA_app, imgsB)

            # l = self.loss(imgsA, recon_A, warped_BA, flow_BA)
            mse = L_mse(imgsA, recon_A)
            l = mse
        return {"loss": l.detach().cpu().numpy()}


class GVSLTrainer_recon_only_with_spatial(GVSLTrainer_do_spatial):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 2
        self.initial_lr = 1e-4
        self.num_iterations_per_epoch = 100

    def build_architecture_and_adaptation_plan(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        backbone = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
        )
        architecture = GVSLArchitecture_recon_only(backbone, num_input_channels)
        raise NotImplementedError("Missing adaptation plan")
        return architecture

    @override
    def train_step(self, batch: dict) -> dict:
        imgsA = batch["imgsA"]
        imgsA_app = batch["imgsA_app"]
        imgsB = batch["imgsB"]

        imgsA = imgsA.to(self.device, non_blocking=True)
        imgsA_app = imgsA_app.to(self.device, non_blocking=True)
        imgsB = imgsB.to(self.device, non_blocking=True)

        with torch.device(self.device):
            # For some reason, the official implementation includes affine transformations and deformations as
            # data augmentations. Mentioned nowhere in the paper...
            # These augmentations benefit from GPU acceleration, and since batchgenerators does not provide GPU support
            # for their transforms, they have to be conducted here
            if self.do_spatial_aug:
                affine_mat, flow = self.spatial_transforms.get_rand_spatial(
                    self.batch_size, self.config_plan.patch_size
                )
                imgsA = self.spatial_transforms.augment_spatial(imgsA, affine_mat, flow)
                imgsA_app = self.spatial_transforms.augment_spatial(imgsA_app, affine_mat, flow)
                imgsB = self.spatial_transforms.augment_spatial(imgsB, affine_mat, flow)

            # self.visualize_brain_slices(imgsA, "imgsA_no_aug.png")
            # self.visualize_brain_slices(imgsB, "imgsB_no_aug.png")
            # return {"loss": np.array(1)}

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                recon_A = self.network(imgsA_app, imgsB)

            # NCC loss tends to get NANs with float16, thus we will not use autocast for loss calculation
            # l = self.loss(imgsA, recon_A, warped_BA, flow_BA)
            mse = L_mse(imgsA, recon_A)
            self.ncc_losses.append(0)
            self.mse_losses.append(mse.detach().item())
            self.smooth_losses.append(0)
            l = mse

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

    def validation_step(self, batch: dict) -> dict:
        imgsA = batch["imgsA"]
        imgsA_app = batch["imgsA_app"]
        imgsB = batch["imgsB"]

        imgsA = imgsA.to(self.device, non_blocking=True)
        imgsA_app = imgsA_app.to(self.device, non_blocking=True)
        imgsB = imgsB.to(self.device, non_blocking=True)

        with torch.no_grad(), torch.device(self.device):
            if self.do_spatial_aug:
                affine_mat, flow = self.spatial_transforms.get_rand_spatial(
                    self.batch_size, self.config_plan.patch_size
                )
                imgsA = self.spatial_transforms.augment_spatial(imgsA, affine_mat, flow)
                imgsA_app = self.spatial_transforms.augment_spatial(imgsA_app, affine_mat, flow)
                imgsB = self.spatial_transforms.augment_spatial(imgsB, affine_mat, flow)

            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                recon_A = self.network(imgsA_app, imgsB)

            # l = self.loss(imgsA, recon_A, warped_BA, flow_BA)
            mse = L_mse(imgsA, recon_A)
            l = mse
        return {"loss": l.detach().cpu().numpy()}
