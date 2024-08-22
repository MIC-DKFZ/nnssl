import os
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from nnssl.architectures.build_architecture import build_network_architecture
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.dataloading.model_genesis_transform import ModelGenesisTransform

from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
from torch import autocast
from nnssl.utilities.helpers import dummy_context

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor


class ModelGenesisTrainer(AbstractBaseTrainer):

    def initialize(self):
        if not self.was_initialized:
            self.network = self.build_architecture(
                self.config_plan, self.num_input_channels, self.num_output_channels
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file("Using torch.compile...")
                self.network = torch.compile(self.network)
                self.print_to_log_file("Compile done.")

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank], find_unused_parameters=True)

            self.loss = self.build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return torch.nn.MSELoss()

    def build_architecture(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        architecture = build_network_architecture(config_plan, num_input_channels, num_output_channels)
        return architecture

    def get_dataloaders(self):
        """
        Dataloader creation is very different depending on the use-case of training.
        This method has to be implemneted for other use-cases aside from MAE more specifically."""
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.config_plan.patch_size

        tr_transforms = self.get_training_transforms()
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size=patch_size)

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

    def train_step(self, batch: dict) -> dict:
        in_data = batch["input"]
        target = batch["target"]
        in_data = in_data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(in_data)
            # del data
            l = self.loss(target, output)

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
        in_data = batch["input"]
        target = batch["target"]
        in_data = in_data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                output = self.network(in_data)
                # del data
                l = self.loss(target, output)

        return {"loss": l.detach().cpu().numpy()}

    def run_training(self):
        self.on_train_start()
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_batch = next(self.dataloader_val)
                    val_outputs.append(self.validation_step(val_batch))
                self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()
        self.on_train_end()

    @staticmethod
    def get_training_transforms() -> AbstractTransform:
        tr_transforms = []

        tr_transforms.append(ModelGenesisTransform())
        tr_transforms.append(NumpyToTensor(["input", "target"], "float"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    @staticmethod
    def get_validation_transforms() -> AbstractTransform:
        return ModelGenesisTrainer.get_training_transforms()


class ModelGenesisTrainer_BS6(ModelGenesisTrainer):

    def __init__(
            self,
            plan: Plan,
            configuration_name: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device("cuda"),
        ):
        plan.configurations[configuration_name].batch_size = 6
        super().__init__(plan, configuration_name, fold, dataset_json, unpack_dataset, device)
