from typing import Union
import torch
from nnssl.architectures.spark_model import SparK3D
from nnssl.architectures.spark_utils import convert_to_spark_cnn

from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.training.loss.spark_loss import SparkLoss
from nnssl.training.lr_scheduler.polylr import PolyLRScheduler
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer
from torch import nn

import valohai
from valohai.config import is_running_in_valohai
from torch import autocast
from nnssl.utilities.helpers import dummy_context
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from nnssl.architectures import spark_utils

from torch._dynamo import OptimizedModule


class SparkMAETrainer(BaseMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_json, pretrain_dataset, device)
        self.mask_percentage: float = 0.75
        self.loss: SparkLoss
        self.stop_at_nans = True
        self.use_mask_token: bool = True
        self.network: SparK3D = ...

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here
        :return:
        """

        return SparkLoss()

    def build_architecture(self, *args, **kwargs) -> nn.Module:
        n_stages = 6
        network = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
            num_classes=1,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
        )

        spark_architecture = convert_to_spark_cnn(network.encoder)
        network.encoder = spark_architecture
        actual_network = SparK3D(network, (160, 160, 160), self.use_mask_token)

        return actual_network

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        data = data.to(self.device, non_blocking=True)
        target = data

        mask = self.mask_creation(self.batch_size, self.config_plan.patch_size, self.mask_percentage).to(
            self.device, non_blocking=True
        )
        spark_utils._cur_active = mask
        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabledq.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(prediction=output, groundtruth=target, mask=mask)
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

    def save_checkpoint(self, filename: str, live_upload: bool = False) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module.architecture
                else:
                    mod = self.network.architecture
                if isinstance(mod, OptimizedModule):
                    mod = mod.architecture._orig_mod

                if self.is_ddp:
                    spk = self.network.module
                else:
                    spk = self.network
                if isinstance(mod, OptimizedModule):
                    spk = mod._orig_mod

                checkpoint = {
                    "network_weights": mod.state_dict(),
                    "spark_weights": spk.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    "logging": self.logger.get_checkpoint(),
                    "_best_ema": self._best_ema,
                    "current_epoch": self.current_epoch + 1,
                    "init_args": self.my_init_kwargs,
                    "trainer_name": self.__class__.__name__,
                }
                torch.save(checkpoint, filename)
                if is_running_in_valohai() and live_upload:
                    filename = f"ckpt_{self.current_epoch}.pth"
                    out_path = valohai.outputs().path(filename)
                    torch.save(checkpoint, out_path)
                    valohai.outputs().live_upload(filename)
            else:
                self.print_to_log_file("No checkpoint written, checkpointing is disabled")

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint["spark_weights"].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith("module."):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint["init_args"]

        self.current_epoch = checkpoint["current_epoch"]
        min_epoch = self.logger.load_checkpoint(checkpoint["logging"])
        # Apparently the val log is not written correctly when we currently save the checkpoint.
        self.current_epoch = min_epoch
        self._best_ema = checkpoint["_best_ema"]

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None:
            if checkpoint["grad_scaler_state"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])

    def validation_step(self, batch: dict) -> dict:
        with torch.no_grad():
            data = batch["data"]
            data = data.to(self.device, non_blocking=True)
            target = data

            mask = self.mask_creation(self.batch_size, self.config_plan.patch_size, self.mask_percentage).to(
                self.device, non_blocking=True
            )
            spark_utils._cur_active = mask
            # Autocast is a little bitch.
            # If the device_type is 'cpu' then it's slow as heck and needs to be disabledq.
            # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
            # So autocast will only be active if we have a cuda device.
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                output = self.network(data)
                # del data
                l = self.loss(prediction=output, groundtruth=target, mask=mask)
            return {"loss": l.detach().cpu().numpy()}

    def log_qualitative_reconstruction_step(
        self,
    ):
        """For each sample in the validation dataloader,"""
        with torch.no_grad():
            for batch_id in range(len(self.recon_dataloader)):
                data = self.recon_dataloader[batch_id]["data"]
                data = data.to(self.device, non_blocking=True)

                mask = self.mask_creation(
                    1,
                    self.config_plan.patch_size,
                    self.mask_percentage,
                    rng_seed=123 + batch_id,
                ).to(self.device, non_blocking=True)
                spark_utils._cur_active = mask

                # Make the mask the same size as the data
                rep_D, rep_H, rep_W = (
                    data.shape[2] // mask.shape[2],
                    data.shape[3] // mask.shape[3],
                    data.shape[4] // mask.shape[4],
                )
                full_mask = (
                    mask.repeat_interleave(rep_D, dim=2)
                    .repeat_interleave(rep_H, dim=3)
                    .repeat_interleave(rep_W, dim=4)
                )

                with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                    reconstruction = self.network(data)  # Doesn't need to be masked as it happens inside.

                    l = [
                        self.loss(reconstruction[i : i + 1], data[i : i + 1], mask[i : i + 1])
                        for i in range(reconstruction.shape[0])
                    ]
                    self.log_img_slices(data, reconstruction, full_mask, l, batch_id)

        return


class SparkMAETrainer5ep(SparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset, device)
        self.num_epochs = 5


class SparkMAETrainer2k(SparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset, device)
        self.num_epochs = 2000


class SparkMAETrainer4k(SparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset, device)
        self.num_epochs = 4000


class SparkMAETrainer5epBS10(SparkMAETrainer5ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # plan.configurations[configuration_name].batch_size = 10
        plan.configurations[configuration_name].batch_size = 10
        print(f"Pre Batch size: {plan.configurations[configuration_name].batch_size}")
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset, device)
        print(f"Post Init Batch size: {self.batch_size}")


class SparkMAETrainer5epBS8(SparkMAETrainer5ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 8
        print(f"Pre Batch size: {plan.configurations[configuration_name].batch_size}")
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset, device)
        print(f"Post Init Batch size: {self.batch_size}")


class SparkMAETrainer5epBS6(SparkMAETrainer5ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 6
        print(f"Pre Batch size: {plan.configurations[configuration_name].batch_size}")
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset, device)
        print(f"Post Init Batch size: {self.batch_size}")


class SparkMAETrainer5epBS4(SparkMAETrainer5ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 4
        print(f"Pre Batch size: {plan.configurations[configuration_name].batch_size}")
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset, device)
        print(f"Post Init Batch size: {self.batch_size}")


class SparkMAETrainer5epBS2(SparkMAETrainer5ep):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 2
        print(f"Pre Batch size: {plan.configurations[configuration_name].batch_size}")
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset, device)
        print(f"Post Init Batch size: {self.batch_size}")


class SparkMAETrainerBS8(SparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 8
        print(f"Pre Batch size: {plan.configurations[configuration_name].batch_size}")
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset, device)
        print(f"Post Init Batch size: {self.batch_size}")


class SparkMAETrainer_test_mask(SparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 1
        print(f"Pre Batch size: {plan.configurations[configuration_name].batch_size}")
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,, device)
        print(f"Post Init Batch size: {self.batch_size}")


class SparkMAETrainer_test_no_mask(SparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 1
        print(f"Pre Batch size: {plan.configurations[configuration_name].batch_size}")
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.use_mask_token = False
        print(f"Post Init Batch size: {self.batch_size}")


class SparkMAETrainerBS7(SparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 7
        print(f"Pre Batch size: {plan.configurations[configuration_name].batch_size}")
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        print(f"Post Init Batch size: {self.batch_size}")


class SparkMAETrainerBS7_noMaskToken(SparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 7
        print(f"Pre Batch size: {plan.configurations[configuration_name].batch_size}")
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.use_mask_token = False
        print(f"Post Init Batch size: {self.batch_size}")


class SparkMAETrainerBS4(SparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 4
        print(f"Pre Batch size: {plan.configurations[configuration_name].batch_size}")
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        print(f"Post Init Batch size: {self.batch_size}")


class SparkMAETrainerBS4_2k(SparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 4
        print(f"Pre Batch size: {plan.configurations[configuration_name].batch_size}")
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        print(f"Post Init Batch size: {self.batch_size}")
        self.num_epochs = 2000


class SparkMAETrainerBS2(SparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 2
        print(f"Pre Batch size: {plan.configurations[configuration_name].batch_size}")
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        print(f"Post Init Batch size: {self.batch_size}")


class SparkMAETrainerBS2_4k(SparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 2
        print(f"Pre Batch size: {plan.configurations[configuration_name].batch_size}")
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.num_epochs = 4000
        print(f"Post Init Batch size: {self.batch_size}")


class SparkMAETrainerBS2_lr5e_2(SparkMAETrainerBS2):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.initial_lr = 5e-2


class SparkMAETrainerBS2_lr1e_1(SparkMAETrainerBS2):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.initial_lr = 1e-1


class SparkMAETrainerBS2_AdamW_1e_3(SparkMAETrainerBS2):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.initial_lr = 1e-3
        self.weight_decay = 1e-2

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class SparkMAETrainerBS2_AdamW_5e_3(SparkMAETrainerBS2):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.initial_lr = 5e-3
        self.weight_decay = 1e-2

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class SparkMAETrainerBS2_AdamW_1e_2(SparkMAETrainerBS2):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.initial_lr = 1e-2
        self.weight_decay = 1e-2

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class SparkMAETrainerBS7_lr_3e2(SparkMAETrainerBS7):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.initial_lr = 3e-2


class SparkMAETrainerBS7_lr_5e2(SparkMAETrainerBS7):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.initial_lr = 5e-2


class SparkMAETrainer_5ep_BS6_mask60(SparkMAETrainer5ep):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 6
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.mask_percentage: float = 0.6


class SparkMAETrainer_5ep_BS7_mask60(SparkMAETrainer5ep):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 7
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.mask_percentage: float = 0.6


class SparkMAETrainer_BS6_250ep(SparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 6
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.num_epochs = 250


class SparkMAETrainer_BS6_500ep(SparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 6
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.num_epochs = 500


class SparkMAETrainer_BS6_1000ep(SparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 6
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.num_epochs = 1000


class SparkMAETrainer_BS6_2000ep(SparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 6
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.num_epochs = 2000


class SparkMAETrainer_BS6_4000ep(SparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 6
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.num_epochs = 4000


class SparkMAETrainer_BS6_1000ep_Mask30(SparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 6
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.num_epochs = 1000
        self.mask_percentage: float = 0.30


class SparkMAETrainer_BS6_1000ep_Mask45(SparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 6
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.num_epochs = 1000
        self.mask_percentage: float = 0.45


class SparkMAETrainer_BS6_1000ep_Mask60(SparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 6
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.num_epochs = 1000
        self.mask_percentage: float = 0.60


class SparkMAETrainer_BS6_1000ep_Mask90(SparkMAETrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        pretrain_dataset: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 6
        super().__init__(plan, configuration_name, fold, dataset_json, pretrain_dataset,device)
        self.num_epochs = 1000
        self.mask_percentage: float = 0.90
