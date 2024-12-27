import torch
from nnssl.training.loss.mse_loss import MAESSIMLoss, MAE_MS_SSIMLoss, MAEL1Loss
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import (
    BaseMAETrainer_BS1,
    BaseMAETrainer_BS2,
    BaseMAETrainer_BS8_1000ep,
)


class BaseMAETrainer_BS2_SSIM(BaseMAETrainer_BS2):

    def __init__(
        self,
        plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        # self.initial_lr = 1e-3

    def build_loss(self):
        return MAESSIMLoss()


class BaseMAETrainer_BS1_MSSSIM(BaseMAETrainer_BS1):
    def build_loss(self):
        return MAE_MS_SSIMLoss()


class BaseMAETrainer_BS1_L1(BaseMAETrainer_BS1):
    def build_loss(self):
        return MAEL1Loss()


class BaseMAETrainer_BS8_ep1000_SSIM(BaseMAETrainer_BS8_1000ep):
    def build_loss(self):
        return MAESSIMLoss()


class BaseMAETrainer_BS8_ep1000_MSSSIM(BaseMAETrainer_BS8_1000ep):
    def build_loss(self):
        return MAE_MS_SSIMLoss()


class BaseMAETrainer_BS8_ep1000_L1(BaseMAETrainer_BS8_1000ep):
    def build_loss(self):
        return MAEL1Loss()
