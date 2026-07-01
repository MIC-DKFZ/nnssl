import os
from typing import Tuple
from loguru import logger
import torch
from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.ssl_data.dataloading.data_loader_3d import nnsslDataLoader3D
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer_BS8_1000ep, BaseMAETrainer_BS1
from random import sample


class BaseMAETrainer_BS8_ep1000_Dataset_1div4(BaseMAETrainer_BS8_1000ep):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.fraction = 1 / 4

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...]):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # Reduce dataset size artificially
        img_ids = dataset_tr.image_identifiers
        n_imgs = len(img_ids)
        img_ids = sample(img_ids, int(n_imgs * self.fraction))
        img_id_set = set(img_ids)
        dataset_tr.image_identifiers = img_ids
        dataset_tr.image_dataset = {k: v for k, v in dataset_tr.image_dataset.items() if k in img_id_set}

        self.print_to_log_file(f"Reduced dataset size from {n_imgs} to {len(img_ids)}")
        logger.info(f"Reduced dataset size from {n_imgs} to {len(img_ids)}")

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


class BaseMAETrainer_BS8_ep1000_Dataset_1div16(BaseMAETrainer_BS8_ep1000_Dataset_1div4):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.fraction = 1 / 16


class BaseMAETrainer_BS8_ep1000_Dataset_1div64(BaseMAETrainer_BS8_ep1000_Dataset_1div4):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.fraction = 1 / 64


class BaseMAETrainer_BS8_ep1000_Dataset_1div256(BaseMAETrainer_BS8_ep1000_Dataset_1div4):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.fraction = 1 / 256


class BaseMAETrainer_BS8_ep1000_Dataset_1div1024(BaseMAETrainer_BS8_ep1000_Dataset_1div4):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.fraction = 1 / 1024
