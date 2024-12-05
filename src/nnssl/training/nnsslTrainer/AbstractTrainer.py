from abc import ABC, abstractmethod
from dataclasses import asdict
from functools import partial
import inspect
import json
from multiprocessing import Pool
import os
from random import sample
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List
from loguru import logger
from tqdm import tqdm
from valohai.config import is_running_in_valohai
from nnssl.configuration import default_num_processes
import signal

import valohai

import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose

from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, isfile, save_json, maybe_mkdir_p, load_json
from torch._dynamo import OptimizedModule


from nnssl.data.raw_dataset import Collection, Dataset, IndependentImage
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.paths import nnssl_preprocessed, nnssl_results
from nnssl.ssl_data.configure_basic_dummyDA import configure_rotation_dummyDA_mirroring_and_inital_patch_size
from nnssl.ssl_data.dataloading.data_loader_3d import nnsslDataLoader3D
from nnssl.ssl_data.dataloading.utils import get_subject_identifiers
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
from valohai.config import is_running_in_valohai

from nnssl.training.dataloading.dataset import nnSSLDatasetBlosc2
from nnssl.training.logging.nnssl_logger import nnSSLLogger
from nnssl.training.lr_scheduler.polylr import PolyLRScheduler
from nnssl.utilities.serialization import make_serializable
from nnssl.utilities.collate_outputs import collate_outputs
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnssl.utilities.helpers import empty_cache
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


def verify_img(img_id, dataset_dir: str, image_dataset: dict[str, IndependentImage]):
    try:
        data, anon, anat, properties = nnSSLDatasetBlosc2.load_case(dataset_dir, image_dataset, img_id)
    except Exception as e:
        return "not_readable"
    if np.isinf(data[:]).any() or np.isnan(data[:]).any():
        return "inf_or_nan"
    return "ok"


class AbstractBaseTrainer(ABC):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # From https://grugbrain.dev/. Worth a read ya big brains ;-)
        # apex predator of grug is complexity
        # complexity bad
        # say again:
        # complexity very bad
        # you say now:
        # complexity very, very bad
        # given choice between complexity or one on one against t-rex, grug take t-rex: at least grug see t-rex
        # complexity is spirit demon that enter codebase through well-meaning but ultimately very clubbable non grug-brain developers and project managers who not fear complexity spirit demon or even know about sometime
        # one day code base understandable and grug can get work done, everything good!
        # next day impossible: complexity demon spirit has entered code and very dangerous situation!

        # OK OK I am guilty. But I tried.
        # https://www.osnews.com/images/comics/wtfm.jpg
        # https://i.pinimg.com/originals/26/b2/50/26b250a738ea4abc7a5af4d42ad93af0.jpg

        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()

        self.device = device

        # ---------------------- print what device we are using ---------------------- #
        if self.is_ddp:  # implicitly it's clear that we use cuda in this case
            print(
                f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
                f"{dist.get_world_size()}."
                f"Setting device to {self.device}"
            )
            self.device = torch.device(type="cuda", index=self.local_rank)
        else:
            if self.device.type == "cuda":
                # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
                self.device = torch.device(type="cuda", index=0)
            print(f"Using device: {self.device}")

        # loading and saving this class for continuing from checkpoint should not happen based on pickling. This
        # would also pickle the network etc. Bad, bad. Instead we just reinstantiate and then load the checkpoint we
        # need. So let's save the init args
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]
        self.my_init_kwargs = make_serializable(self.my_init_kwargs)
        # ------ Saving all the init args into class variables for later access ------ #
        self.plan: Plan = plan
        self.config_plan: ConfigurationPlan = plan.configurations[configuration_name]
        self.configuration_name = configuration_name
        self.pretrain_json = pretrain_json
        self.fold = fold
        if is_running_in_valohai():
            self.current_epoch_log = {}

        # ----------------------- Setting all the folder names. ---------------------- #
        ###  We need to make sure things don't crash in case we are just running
        # inference and some of the folders may not be defined!
        self.preprocessed_dataset_folder_base = (
            join(nnssl_preprocessed, self.plan.dataset_name) if nnssl_preprocessed is not None else None
        )
        self.output_folder_base = (
            join(
                nnssl_results,
                self.plan.dataset_name,
                self.__class__.__name__ + "__" + self.plan.plans_name + "__" + configuration_name,
            )
            if nnssl_results is not None
            else None
        )
        self.output_folder = join(self.output_folder_base, f"fold_{fold}")

        self.preprocessed_dataset_folder = join(
            self.preprocessed_dataset_folder_base, self.config_plan.data_identifier
        )
        # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
        # be a different configuration in the same plans
        # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
        # "previous_stage" and "next_stage"). Otherwise it won't work!

        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.momentum = 0.99
        self.nesterov = True
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 1000
        self.current_epoch = 0

        ### Dealing with labels/regions
        self.num_input_channels = 1  # -> self.initialize()
        self.num_output_channels = 1  # Assign later depending on the ssl training scheme.
        self.network = None  # -> self._get_network()
        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        self.grad_scaler = GradScaler() if self.device.type == "cuda" else None
        self.loss = None  # -> self.initialize

        self.stop_at_nans = False

        ### Simple logging. Don't take that away from me!
        # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
        # logging
        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(
            self.output_folder,
            "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt"
            % (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second),
        )
        self.logger = nnSSLLogger()

        ### placeholders
        self.dataloader_train = self.dataloader_val = None  # see on_train_start

        ### initializing stuff for remembering things and such
        self._best_ema = None

        ### checkpoint saving stuff
        self.save_every = 50
        self.disable_checkpointing = False

        ## DDP batch size and oversampling can differ between workers and needs adaptation
        # we need to change the batch size in DDP because we don't use any of those distributed samplers
        # Todo: Should likely be moved to initialize() call, since it's not needed during init at all.
        #   This also allows overriding previous batch_size settings easily.
        self._set_batch_size()

        self.was_initialized = False

        self.exit_training_flag = False # This is a signal flag that can be raised to exit gracefully
        self.print_to_log_file(
            "\n#######################################################################\n"
            "Please cite the following paper when using nnU-Net:\n"
            "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
            "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
            "Nature methods, 18(2), 203-211.\n"
            "#######################################################################\n",
            also_print_to_console=True,
            add_timestamp=False,
        )

    def _set_batch_size(self):
        if not self.is_ddp:
            # set batch size to what the plan says, leave oversample untouched
            logger.info(f"Not using DDP. Setting batch size for single gpu to {self.config_plan.batch_size}.")
            self.batch_size = self.config_plan.batch_size
        else:
            # batch size is distributed over DDP workers and we need to change oversample_percent for each worker
            batch_sizes = []

            world_size = dist.get_world_size()
            my_rank = dist.get_rank()
            logger.info(
                f"Using DDP. Total Batch size {self.config_plan.batch_size} distributed across all {world_size} gpus."
            )

            global_batch_size = self.config_plan.batch_size

            assert global_batch_size >= world_size, (
                f"Cannot run DDP if the batch size ({global_batch_size}) is smaller than the number of GPUs ({world_size})... Duh."
            )

            batch_size_per_GPU = np.ceil(global_batch_size / world_size).astype(int)

            for rank in range(world_size):
                if (rank + 1) * batch_size_per_GPU > global_batch_size:
                    batch_size = batch_size_per_GPU - ((rank + 1) * batch_size_per_GPU - global_batch_size)
                else:
                    batch_size = batch_size_per_GPU

                batch_sizes.append(batch_size)

            print("worker", my_rank, "batch_size", batch_sizes[my_rank])
            # self.print_to_log_file("worker", my_rank, "oversample", oversample_percents[my_rank])
            # self.print_to_log_file("worker", my_rank, "batch_size", batch_sizes[my_rank])

            self.batch_size = batch_sizes[my_rank]

    @abstractmethod
    def build_architecture(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int, *args, **kwargs
    ) -> torch.nn.Module:
        pass

    @abstractmethod
    def build_loss(self):
        pass

    @abstractmethod
    def train_step(self, batch: dict) -> dict:
        pass

    @abstractmethod
    def validation_step(self, batch: dict) -> dict:
        pass

    def initialize(self):
        if not self.was_initialized:
            self.network = self.build_architecture(
                self.config_plan, self.num_input_channels, self.num_output_channels
            ).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file("Using torch.compile...")
                self.network = torch.compile(self.network)

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

    def exit_training(self):
        self.exit_training_flag = True

    def run_training(self):
        try:
            self.on_train_start()

            for epoch in range(self.current_epoch, self.num_epochs):
                self.on_epoch_start()

                self.on_train_epoch_start()
                train_outputs = []

                for batch_id in tqdm(
                    range(self.num_iterations_per_epoch),
                    desc=f"Epoch {epoch}",
                    disable=True if (("LSF_JOBID" in os.environ) or is_running_in_valohai()) else False,
                ):
                    train_outputs.append(self.train_step(next(self.dataloader_train)))

                self.on_train_epoch_end(train_outputs)

                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = []
                    for batch_id in range(self.num_val_iterations_per_epoch):
                        val_outputs.append(self.validation_step(next(self.dataloader_val)))
                    self.on_validation_epoch_end(val_outputs)

                self.on_epoch_end()
                if self.raise_reschedule_flag:
                    # This is a signal that we need to resubmit, so we break the loop and exit gracefully
                    raise KeyboardInterrupt
            self.on_train_end()
        except KeyboardInterrupt:
            self.print_to_log_file("Keyboard interrupt. Exiting gracefully.")
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
            raise KeyboardInterrupt

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = (f"{dt_object}:", *args)

            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, "a+") as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print(f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info())
                    # sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)

    def print_plans(self):
        if self.local_rank == 0:
            dct = deepcopy(asdict(self.plan))
            del dct["configurations"]
            self.print_to_log_file(
                f"\nThis is the configuration used by this "
                f"training:\nConfiguration name: {self.configuration_name}\n",
                asdict(self.config_plan),
                "\n",
                add_timestamp=False,
            )
            self.print_to_log_file("These are the global plan.json settings:\n", dct, "\n", add_timestamp=False)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def get_tr_and_val_datasets(self):
        # create dataset split (We only have 'all' as splits anyway!)
        tr_subjects, val_subjects = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        collection = Collection.from_dict(self.pretrain_json)
        dataset_tr = nnSSLDatasetBlosc2(self.preprocessed_dataset_folder, collection, tr_subjects)
        dataset_val = nnSSLDatasetBlosc2(self.preprocessed_dataset_folder, collection, val_subjects)

        valid_images = load_json(join(self.preprocessed_dataset_folder_base, "valid_imgs.json"))
        valid_image_ids = [i["image_name"] for i in valid_images]

        tr_imgs_removed = self.keep_valid(valid_image_ids, dataset_tr)
        logger.info(f"Removed {tr_imgs_removed} broken images from train dataset.")
        logger.info(f"Number of training images: {len(dataset_tr.image_identifiers)}")
        vl_imgs_removed = self.keep_valid(valid_image_ids, dataset_val)
        logger.info(f"Removed {vl_imgs_removed} broken images from val dataset.")
        logger.info(f"Number of validation images: {len(dataset_val.image_identifiers)}")

        return dataset_tr, dataset_val

    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.config_plan.patch_size
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)
        if do_dummy_2d_data_aug:
            self.print_to_log_file("Using dummy 2D data augmentation")

        # ------------------------ Training data augmentations ----------------------- #
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            mirror_axes,
            do_dummy_2d_data_aug,
            order_resampling_data=3,
            order_resampling_seg=1,
        )

        # ----------------------- Validation data augmentations ---------------------- #
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size)

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

    def interrupt_at_nans(self, losses: list[dict]):
        if self.stop_at_nans:
            threshold = 20
            nans = sum([1 if np.isnan(l["loss"]) else 0 for l in losses])
            if nans > threshold:
                raise RuntimeError(f"More than {threshold} NaN's detected in loss. Aborting.")

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...]):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnsslDataLoader3D(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
        )
        dl_val = nnsslDataLoader3D(
            dataset_val,
            self.batch_size,
            self.config_plan.patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
        )
        return dl_tr, dl_val

    @staticmethod
    @abstractmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
    ) -> AbstractTransform:
        pass

    @staticmethod
    @abstractmethod
    def get_validation_transforms() -> AbstractTransform:
        pass

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        self.print_plans()
        empty_cache(self.device)

        if self.is_ddp:
            dist.barrier()

        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()
        # Guarantee to only use data that is readable and not inf or nan

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(asdict(self.plan), join(self.output_folder_base, "plans.json"), sort_keys=False)

        self._save_debug_information()

    def on_train_end(self):
        # dirty hack because on_epoch_end increments the epoch counter and this is executed afterwards.
        # This will lead to the wrong current epoch to be stored
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        # now we can delete latest
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, "w") as f:
            sys.stdout = f
            if self.dataloader_train is not None:
                self.dataloader_train._finish()
            if self.dataloader_val is not None:
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def on_train_epoch_end(self, train_outputs: List[dict]):
        self.interrupt_at_nans(train_outputs)
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs["loss"])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs["loss"])
        self.logger.log("train_losses", loss_here, self.current_epoch)

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)

        if self.is_ddp:
            world_size = dist.get_world_size()
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated["loss"])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated["loss"])
        self.logger.log("val_losses", loss_here, self.current_epoch)

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log("lrs", self.optimizer.param_groups[0]["lr"], self.current_epoch)

    def on_validation_epoch_start(self):
        self.network.eval()

    def on_epoch_start(self):
        self.logger.log("epoch_start_timestamps", time(), self.current_epoch)

    def on_epoch_end(self):
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)

        self.print_to_log_file(
            "train_loss", np.round(self.logger.my_fantastic_logging["train_losses"][-1], decimals=4)
        )
        self.print_to_log_file("val_loss", np.round(self.logger.my_fantastic_logging["val_losses"][-1], decimals=4))
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s"
        )
        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        # handle 'best' checkpointing. val_loss smaller than best_ema
        if self._best_ema is None or self.logger.my_fantastic_logging["val_losses"][-1] < self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging["val_losses"][-1]
            self.print_to_log_file(f"Yayy! New best val loss: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))

        if self.local_rank == 0:
            if self.current_epoch % 50 == 0:
                self.print_to_log_file("Saving checkpoint...")
                self.save_checkpoint(
                    join(self.output_folder, f"checkpoint_epoch_{self.current_epoch}.pth"), live_upload=True
                )
            self.logger.plot_progress_png(self.output_folder)

        if is_running_in_valohai():
            self.current_epoch_log["epoch"] = int(self.current_epoch)
            self.current_epoch_log["train_loss"] = float(self.logger.my_fantastic_logging["train_losses"][-1])
            self.current_epoch_log["val_loss"] = float(self.logger.my_fantastic_logging["val_losses"][-1])
            self.current_epoch_log["learning_rate"] = float(self.logger.my_fantastic_logging["lrs"][-1])
            self.current_epoch_log["epoch_time"] = float(
                np.round(
                    self.logger.my_fantastic_logging["epoch_end_timestamps"][-1]
                    - self.logger.my_fantastic_logging["epoch_start_timestamps"][-1],
                    decimals=2,
                )
            )
            print(json.dumps(self.current_epoch_log))
            self.current_epoch_log = {}

        self.current_epoch += 1

    def save_checkpoint(self, filename: str, live_upload: bool = False) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    "network_weights": mod.state_dict(),
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
        for k, value in checkpoint["network_weights"].items():
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

    def perform_actual_validation(self, save_probabilities: bool = False):
        print("Actual Validation is trainer specific and needs to be written here. To be implemented late!")

    def _do_i_compile(self):
        return ("nnUNet_compile" in os.environ.keys()) and (
            os.environ["nnUNet_compile"].lower() in ("true", "1", "t")
        )

    def _save_debug_information(self):
        # saving some debug information
        if self.local_rank == 0:
            dct = {}
            for k in self.__dir__():
                if not k.startswith("__"):
                    if not callable(getattr(self, k)) or k in [
                        "loss",
                    ]:
                        dct[k] = str(getattr(self, k))
                    elif k in [
                        "network",
                    ]:
                        dct[k] = str(getattr(self, k).__class__.__name__)
                    else:
                        # print(k)
                        pass
                if k in ["dataloader_train", "dataloader_val"]:
                    if hasattr(getattr(self, k), "generator"):
                        dct[k + ".generator"] = str(getattr(self, k).generator)
                    if hasattr(getattr(self, k), "num_processes"):
                        dct[k + ".num_processes"] = str(getattr(self, k).num_processes)
                    if hasattr(getattr(self, k), "transform"):
                        dct[k + ".transform"] = str(getattr(self, k).transform)
            import subprocess

            hostname = subprocess.getoutput(["hostname"])
            dct["hostname"] = hostname
            torch_version = torch.__version__
            if self.device.type == "cuda":
                gpu_name = torch.cuda.get_device_name()
                dct["gpu_name"] = gpu_name
                cudnn_version = torch.backends.cudnn.version()
            else:
                cudnn_version = "None"
            dct["device"] = str(self.device)
            dct["torch_version"] = torch_version
            dct["cudnn_version"] = cudnn_version
            save_json(dct, join(self.output_folder, "debug.json"))

    @staticmethod
    def keep_valid(valid_image_names: list[str], dataset: nnSSLDatasetBlosc2):

        # --------------------------- Remove broken images --------------------------- #
        pre_removal_len = len(dataset.image_identifiers)
        dataset.image_dataset = {k: v for k, v in dataset.image_dataset.items() if v.image_name in valid_image_names}
        dataset.image_identifiers = list(dataset.image_dataset.keys())
        post_removal_len = len(dataset.image_identifiers)
        removed_images = pre_removal_len - post_removal_len

        return removed_images

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        # if self.fold == "all":
        # if fold==all then we use all images for training and validation
        # There used to be a if/else for the case that we don't use all samples, but we only do self-supervised thingies,
        #   so we use all samples for training and validation
        splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
        if not isfile(splits_file):
            self.print_to_log_file("Creating new 5-fold cross-validation split...")
            subject_identifiers = get_subject_identifiers(self.preprocessed_dataset_folder)
            assert len(subject_identifiers) != 0, "No subjects found. Aborting"
            all_keys_sorted = sorted(list(np.sort(subject_identifiers)))
            n_val_subjects = min(50, int(len(subject_identifiers) / 100))
            val_subjects = sample(all_keys_sorted, n_val_subjects)
            train_subjects = list(set(all_keys_sorted) - set(val_subjects))
            splits = {"train": list(train_subjects), "val": list(val_subjects)}
            save_json(splits, splits_file)
        else:
            splits = load_json(splits_file)

        tr_subjects = splits["train"]
        val_subjects = splits["val"]
        return tr_subjects, val_subjects
