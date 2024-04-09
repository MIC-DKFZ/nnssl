import torch
from nnssl.architectures.spark_utils import convert_to_spark_cnn

from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.training.loss.spark_loss import SparkLoss
from nnssl.training.nnsslTrainer.BaseMAETrainer import BaseMAETrainer
from torch import nn

from torch import nn
import torch
from torch import autocast
from nnssl.training.nnsslTrainer.SparkTrainer import SparkMAETrainer
from nnssl.utilities.helpers import dummy_context
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from nnssl.architectures import spark_utils


class LoadingEfficientSparkMAETrainer(SparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        self.loading_multiplicator = 2
        self.sub_steps = 4
        batch_size = plan.configurations[configuration_name].batch_size
        self.loading_batch_size = batch_size * self.loading_multiplicator
        self.mask_percentage: float = 0.75

        # Asserts that we load twice the samples to memory, which we then can sub-sample from.
        plan.configurations[configuration_name].batch_size = self.loading_batch_size
        super().__init__(plan, configuration_name, fold, dataset_json, unpack_dataset, device)
        self.loss: SparkLoss

        self.sub_batch_size = batch_size

    def _build_loss(self):
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
        return network

    def train_step(self, batch: dict) -> list[dict]:
        data = batch["data"]
        # data = data.to(self.device, non_blocking=True)
        # target = data

        indices = torch.tensor([i for i in range(data.shape[0])])

        losses = []
        for _ in range(self.sub_steps):

            sub_data = data[torch.permute(indices, dims=(0,))[: self.sub_batch_size]]
            sub_data = sub_data.to(self.device, non_blocking=True)
            sub_target = sub_data

            mask = self.mask_creation(self.sub_batch_size, self.config_plan.patch_size, self.mask_percentage).to(
                self.device, non_blocking=True
            )
            spark_utils._cur_active = mask
            self.optimizer.zero_grad(set_to_none=True)
            # Autocast is a little bitch.
            # If the device_type is 'cpu' then it's slow as heck and needs to be disabledq.
            # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
            # So autocast will only be active if we have a cuda device.
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                output = self.network(sub_data)
                # del data
                l = self.loss(prediction=output, groundtruth=sub_target, mask=mask)
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
            losses.append({"loss": l.detach().cpu().numpy()})
        return losses

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.extend(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.extend(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()

    def validation_step(self, batch: dict) -> dict:
        with torch.no_grad():
            data = batch["data"]
            indices = torch.tensor([i for i in range(data.shape[0])])

            losses = []
            for i in range(self.sub_steps):
                sub_data = data[torch.permute(indices, dims=(0,))[: self.sub_batch_size]]
                sub_data = sub_data.to(self.device, non_blocking=True)
                sub_target = sub_data

                mask = self.mask_creation(self.batch_size, self.config_plan.patch_size, self.mask_percentage).to(
                    self.device, non_blocking=True
                )
                spark_utils._cur_active = mask
                # Autocast is a little bitch.
                # If the device_type is 'cpu' then it's slow as heck and needs to be disabledq.
                # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
                # So autocast will only be active if we have a cuda device.
                with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                    output = self.network(sub_data)
                    # del data
                    l = self.loss(prediction=output, groundtruth=sub_target, mask=mask)
                losses.append({"loss": l.detach().cpu().numpy()})
            return losses

    def log_qualitative_reconstruction_step(
        self,
    ):
        """For each sample in the validation dataloader,"""
        with torch.no_grad():
            for batch_id in range(len(self.recon_dataloader)):
                data = self.recon_dataloader[batch_id]["data"]
                data = data.to(self.device, non_blocking=True)

                mask = self.mask_creation(
                    self.batch_size, self.config_plan.patch_size, self.mask_percentage, rng_seed=123 + batch_id
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


class LoadingEfficientSparkMAETrainer5ep(LoadingEfficientSparkMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 5
