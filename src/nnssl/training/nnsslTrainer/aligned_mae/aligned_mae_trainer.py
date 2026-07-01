import copy
from typing import Union, Tuple, List
from typing_extensions import override

import torch
import numpy as np

from torch import autocast

from nnssl.architectures.consis_arch import ConsisMAE, FeatureContrastiveDecoderAligned
from nnssl.ssl_data.configure_basic_dummyDA import (
    configure_rotation_dummyDA_mirroring_and_inital_patch_size,
)
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import (
    BaseMAETrainer,
)
from nnssl.utilities.helpers import dummy_context
from nnssl.ssl_data.dataloading.aligned_transform import OverlapTransform


class BaseAlignedMAETrainer(BaseMAETrainer):
    """
    Base class for Key-Value Consistency EVA Trainer.
    This class inherits from EvaMAETrainer and is designed to handle
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the BaseKVConsisEvaTrainer with the given arguments.
        """
        super().__init__(*args, **kwargs)

        # Default initial patch size, can be overridden in get_dataloaders
        self.initial_patch_size = (256, 256, 256)
        self.total_batch_size = 2
        self.initial_lr = 1e-2
        self.num_epochs = 250
        self.teacher = None
        self.teacher_mom = 0.995  # Momentum for the teacher model update
        self.config_plan.patch_size = (
            160,
            160,
            160,
        )  # Default patch size for KV Consis Eva

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        # Create the loss function
        return AlignedMAELoss(device=self.device)

    def on_validation_epoch_start(self):
        # self.network.eval()

        # the predictor part of the model requires network to be in training mode...
        pass

    @override
    def build_architecture_and_adaptation_plan(
            self,
            config_plan,
            num_input_channels: int,
            num_output_channels: int,
            *args,
            **kwargs,
    ):
        # ---------------------------- Create architecture --------------------------- #
        architecture = ConsisMAE(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=False,
            only_last_stage_as_latent=False,
            use_projector=False,
        )
        # --------------------- Build associated adaptation plan --------------------- #
        # no changes to original mae since projector can be thrown away
        adapt_plan = self.save_adaption_plan(num_input_channels)
        return architecture, adapt_plan

    def get_validation_transforms(self):
        """
        Returns the validation transforms for the model.
        This method is overridden to provide specific validation transforms.
        """
        return OverlapTransform(
            train="none",
            data_key="data",
            initial_patch_size=self.initial_patch_size,
            patch_size=self.config_plan.patch_size,
        )

    def get_training_transforms(
            self,
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: dict,
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            order_resampling_data: int = 3,
            order_resampling_seg: int = 1,
            border_val_seg: int = -1,
            use_mask_for_norm: List[bool] = None,
    ):
        """
        Returns the training transforms for the model.
        This method is overridden to provide specific training transforms.
        """
        return OverlapTransform(
            train="train",
            data_key="data",
            initial_patch_size=self.initial_patch_size,
            patch_size=patch_size,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            order_resampling_data=order_resampling_data,
            order_resampling_seg=order_resampling_seg,
            border_val_seg=border_val_seg,
            use_mask_for_norm=use_mask_for_norm,
        )

    def get_dataloaders(self):
        """
        Dataloader creation is very different depending on the use-case of training.
        This method has to be implemneted for other use-cases aside from MAE more specifically.
        """
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

        self.initial_patch_size = initial_patch_size
        self.print_to_log_file("Initial patch size: {}".format(initial_patch_size))

        # ------------------------ Training data augmentations ----------------------- #
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            mirror_axes,
            do_dummy_2d_data_aug,
            order_resampling_data=3,
            order_resampling_seg=1,
            use_mask_for_norm=self.config_plan.use_mask_for_norm,
        )

        # ----------------------- Validation data augmentations ---------------------- #
        val_transforms = self.get_validation_transforms()

        return self.make_generators(initial_patch_size, tr_transforms, val_transforms)

    def on_train_start(self):
        super().on_train_start()
        if self.teacher is None:
            # create a deep copy of the network to use as a teacher
            self.teacher = copy.deepcopy(self.network)
            self.teacher = self.teacher.to(self.device)
            self.teacher = (
                self.teacher.eval()
            )  # set the teacher to eval mode and not training
            for param in self.teacher.parameters():
                param.requires_grad = False

    def ema(self, teacher_model, student_model, update_bn=False):
        mom = self.teacher_mom

        if mom == 0.0:
            # if the momentum is 0, we just copy the student model to the teacher model
            teacher_model.load_state_dict(student_model.state_dict())
            return

        for p_s, p_t in zip(student_model.parameters(), teacher_model.parameters()):
            p_t.data = mom * p_t.data + (1 - mom) * p_s.data

        if not update_bn:
            return  # update BN stat buffers if required
        for (n_s, m_s), (n_t, m_t) in zip(
                student_model.named_modules(), teacher_model.named_modules()
        ):
            if isinstance(m_s, torch.nn.modules.batchnorm._NormBase) and n_s == n_t:
                m_t.running_mean.data = (
                        mom * m_t.running_mean.data + (1 - mom) * m_s.running_mean.data
                )
                m_t.running_var.data = (
                        mom * m_t.running_var.data + (1 - mom) * m_s.running_var.data
                )

    def shared_step(self, batch: dict, is_train: bool = True) -> dict:
        """
        Shared step for both training and validation.
        This method is overridden to provide specific shared step logic.
        """
        data, bboxes = batch["all_crops"], batch["rel_bboxes"]

        data = data.to(self.device, non_blocking=True)
        bboxes = bboxes.to(self.device, non_blocking=True)

        with torch.no_grad():
            self.teacher.eval()
            teacher_output = self.teacher(data)
            teacher_output = {
                k: v
                for k, v in teacher_output.items()
                if k == "proj" or k == "image_latent"
            }
            self.network.train()  # set the network to training mode

        # We use the self.batch_size as it is not identical with the plan batch_size in ddp cases.
        mask = self.mask_creation(
            2 * self.batch_size, self.config_plan.patch_size, self.mask_percentage
        ).to(self.device, non_blocking=True)
        # Make the mask the same size as the data
        rep_D, rep_H, rep_W = (
            data.shape[2] // mask.shape[2],
            data.shape[3] // mask.shape[3],
            data.shape[4] // mask.shape[4],
        )
        mask = (
            mask.repeat_interleave(rep_D, dim=2)
            .repeat_interleave(rep_H, dim=3)
            .repeat_interleave(rep_W, dim=4)
        )

        masked_data = data * mask

        if is_train:
            self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            with torch.no_grad() if not is_train else dummy_context():
                output = self.network(masked_data)
                # del data
                loss_dict = self.loss(
                    model_output=output,
                    target=teacher_output,
                    gt_recon=data,
                    rel_bboxes=bboxes,
                    mask=mask,
                )
                l = loss_dict["loss"]

        if is_train:
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

            # update the teacher network with momentum of 0.95
            with torch.no_grad():
                self.ema(self.teacher, self.network, update_bn=False)

        return {k: v.detach().cpu().numpy() for k, v in loss_dict.items()}

    def train_step(self, batch: dict) -> dict:
        return self.shared_step(batch, is_train=True)

    def validation_step(self, batch: dict) -> dict:
        return self.shared_step(batch, is_train=False)


class AlignedMAE128Trainer(BaseAlignedMAETrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_batch_size = 4
        self.config_plan.patch_size = (128, 128, 128)

    @override
    def build_architecture_and_adaptation_plan(
            self,
            config_plan,
            num_input_channels: int,
            num_output_channels: int,
            *args,
            **kwargs,
    ):
        # ---------------------------- Create architecture --------------------------- #
        architecture = ConsisMAE(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=False,
            only_last_stage_as_latent=False,
            use_projector=True,
        )
        # --------------------- Build associated adaptation plan --------------------- #
        # no changes to original mae since projector can be thrown away
        adapt_plan = self.save_adaption_plan(num_input_channels)
        return architecture, adapt_plan


class AlignedMAETrainer(AlignedMAE128Trainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the ConMAETrainer with the given arguments.
        """
        super().__init__(*args, **kwargs)
        self.total_batch_size = 4
        self.teacher_mom = 0.995
        self.initial_lr = 1e-2
        self.num_epochs = 1000
        self.mask_percentage = 0.75  # Default mask percentage for ConMAE
        self.config_plan.patch_size = (128, 128, 128)  # Patch size for ConMAE

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic for ConMAE.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(device=self.device, recon_weight=5.0)

    def build_architecture_and_adaptation_plan(
            self,
            config_plan,
            num_input_channels: int,
            num_output_channels: int,
            *args,
            **kwargs,
    ):
        # ---------------------------- Create architecture --------------------------- #
        architecture = ConsisMAE(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=False,
            only_last_stage_as_latent=False,
            use_projector=True,
        )
        # --------------------- Build associated adaptation plan --------------------- #
        # no changes to original mae since projector can be thrown away
        adapt_plan = self.save_adaption_plan(num_input_channels)
        return architecture, adapt_plan


class AlignedMAELR5Trainer(AlignedMAETrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the ConMAETrainer with the given arguments.
        """
        super().__init__(*args, **kwargs)
        self.initial_lr = 5e-3


class AlignedMAEFTTrainer(AlignedMAE128Trainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the ConMAETrainer with the given arguments.
        """
        super().__init__(*args, **kwargs)
        self.total_batch_size = 2 #4
        self.teacher_mom = 0.995
        self.initial_lr = 1e-2
        self.num_epochs = 250
        self.mask_percentage = 0.75  # Default mask percentage for ConMAE
        self.config_plan.patch_size = (128, 128, 128)  # Patch size for ConMAE

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        # Create the loss function
        return AlignedMAELoss(
            device=self.device, recon_weight=5.0, fg_cos_weight=1.0, ntxent_weight=0.1
        )


class AlignedMAEFTConAnisoTrainer(AlignedMAEFTTrainer):

    def __init__(self, *args, **kwargs):
        super(AlignedMAEFTConAnisoTrainer, self).__init__(*args, **kwargs)
        self.total_batch_size = 4
        self.teacher_mom = 0.995
        self.initial_lr = 1e-2
        self.num_epochs = 250
        self.mask_percentage = 0.75  # Default mask percentage for ConMAE
        self.config_plan.patch_size = (256, 256, 32)

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic for low contrast ConMAE.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(
            out_size=(2, 7, 7),
            device=self.device,
            recon_weight=5.0,
            fg_cos_weight=1.0,
            ntxent_weight=0.1,
            fine_grained_contrastive=False,
            fine_grained_cosine_regression=True,  # fine grained reg
        )


class AlignedMAEFTConAnisoB16Trainer(AlignedMAEFTConAnisoTrainer):

    def __init__(self, *args, **kwargs):
        super(AlignedMAEFTConAnisoB16Trainer, self).__init__(*args, **kwargs)
        self.total_batch_size = 16
        self.config_plan.patch_size = (128, 128, 32)


class AlignedMAEFTNoConTrainer(AlignedMAEFTTrainer):

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic for low contrast ConMAE.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(
            device=self.device, recon_weight=5.0, fg_cos_weight=1.0, ntxent_weight=0.0
        )


class GramAlignedMAEFTNoConTrainer(AlignedMAEFTTrainer):

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic for low contrast ConMAE.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(
            device=self.device,
            recon_weight=5.0,
            fg_cos_weight=2.0,
            ntxent_weight=0.0,
            fine_grained_contrastive=False,
            fine_grained_cosine_regression=False,  # gram reg
        )


class GramAlignedMAEFTNoConNoProjTrainer(GramAlignedMAEFTNoConTrainer):

    def build_architecture_and_adaptation_plan(
            self,
            config_plan,
            num_input_channels: int,
            num_output_channels: int,
            *args,
            **kwargs,
    ):
        # ---------------------------- Create architecture --------------------------- #
        architecture = ConsisMAE(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=False,
            only_last_stage_as_latent=False,
            use_projector=False,
        )
        # --------------------- Build associated adaptation plan --------------------- #
        # no changes to original mae since projector can be thrown away
        adapt_plan = self.save_adaption_plan(num_input_channels)
        return architecture, adapt_plan


class GramAlignedMAEFTConTrainer(AlignedMAEFTTrainer):

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic for low contrast ConMAE.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(
            device=self.device,
            recon_weight=5.0,
            fg_cos_weight=2.0,
            ntxent_weight=0.1,
            fine_grained_contrastive=False,
            fine_grained_cosine_regression=False,  # gram reg
        )


class GramAlignedMAEFTConNoProjTrainer(GramAlignedMAEFTConTrainer):

    def build_architecture_and_adaptation_plan(
            self,
            config_plan,
            num_input_channels: int,
            num_output_channels: int,
            *args,
            **kwargs,
    ):
        # ---------------------------- Create architecture --------------------------- #
        architecture = ConsisMAE(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=False,
            only_last_stage_as_latent=False,
            use_projector=False,
            use_projector_global=True,
        )
        # --------------------- Build associated adaptation plan --------------------- #
        # no changes to original mae since projector can be thrown away
        adapt_plan = self.save_adaption_plan(num_input_channels)
        return architecture, adapt_plan


class GramAlignedMAEFTConNoProjAnisoTrainer(GramAlignedMAEFTConNoProjTrainer):

    def __init__(self, *args, **kwargs):
        super(GramAlignedMAEFTConNoProjAnisoTrainer, self).__init__(*args, **kwargs)
        self.total_batch_size = 4
        self.teacher_mom = 0.995
        self.initial_lr = 1e-2
        self.num_epochs = 250
        self.mask_percentage = 0.75  # Default mask percentage for ConMAE
        self.config_plan.patch_size = (256, 256, 32)

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic for low contrast ConMAE.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(
            out_size=(2, 7, 7),
            device=self.device,
            recon_weight=5.0,
            fg_cos_weight=2.0,
            ntxent_weight=0.1,
            fine_grained_contrastive=False,
            fine_grained_cosine_regression=False,  # gram reg
        )


class GramAlignedMAEFTConNoProjAnisoB16Trainer(GramAlignedMAEFTConNoProjAnisoTrainer):

    def __init__(self, *args, **kwargs):
        super(GramAlignedMAEFTConNoProjAnisoB16Trainer, self).__init__(*args, **kwargs)
        self.total_batch_size = 16
        self.config_plan.patch_size = (128, 128, 32)


class AlignedConConMAETrainer(AlignedMAETrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the ConMAETrainer with the given arguments.
        """
        super().__init__(*args, **kwargs)
        self.total_batch_size = 4
        self.teacher_mom = 0.995
        self.initial_lr = 5e-3
        self.num_epochs = 1000
        self.mask_percentage = 0.75  # Default mask percentage for ConMAE
        self.config_plan.patch_size = (128, 128, 128)  # Patch size for ConMAE

    def build_loss(self):
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(
            device=self.device,
            recon_weight=5.0,
            fg_cos_weight=0.5,
            ntxent_weight=0.1,
            fine_grained_contrastive=True,
        )


class AlignedConConMAEFTTrainer(AlignedMAEFTTrainer):

    def build_loss(self):
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(
            device=self.device,
            recon_weight=5.0,
            fg_cos_weight=0.2,
            ntxent_weight=0.1,
            fine_grained_contrastive=True,
        )


class ConMAETrainer(AlignedConConMAETrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the ConMAETrainer with the given arguments.
        """
        super().__init__(*args, **kwargs)
        self.total_batch_size = 4
        self.teacher_mom = 0.995
        self.initial_lr = 5e-3
        self.num_epochs = 1000
        self.mask_percentage = 0.75  # Default mask percentage for ConMAE
        self.config_plan.patch_size = (128, 128, 128)  # Patch size for ConMAE

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic for ConMAE.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(
            device=self.device,
            recon_weight=5.0,
            fg_cos_weight=0.0,
            ntxent_weight=0.1,
            fine_grained_contrastive=False,
        )


class ConMAEFTTrainer(AlignedMAEFTTrainer):

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic for ConMAE.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(
            device=self.device, recon_weight=5.0, fg_cos_weight=0.0, ntxent_weight=0.1
        )


class ConMAEFTAnisoTrainer(ConMAEFTTrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the ConMAETrainer with the given arguments.
        """
        super().__init__(*args, **kwargs)
        self.total_batch_size = 4
        self.teacher_mom = 0.995
        self.initial_lr = 1e-2
        self.num_epochs = 250
        self.mask_percentage = 0.75  # Default mask percentage for ConMAE
        self.config_plan.patch_size = (256, 256, 32)

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic for ConMAE.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(
            out_size=(2, 7, 7),
            device=self.device,
            recon_weight=5.0,
            fg_cos_weight=0.0,
            ntxent_weight=0.1,
        )


class ConMAEFTAnisoB16Trainer(ConMAEFTAnisoTrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the ConMAETrainer with the given arguments.
        """
        super().__init__(*args, **kwargs)
        self.total_batch_size = 16
        self.config_plan.patch_size = (128, 128, 32)


class FeatConDecAlignedMAEFTTrainer(ConMAETrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the FeatConDecAlignedMAEFTTrainer with the given arguments.
        This class is specifically designed for training models with Feature Contrastive Decoder.
        """
        super().__init__(*args, **kwargs)
        self.total_batch_size = 4
        self.teacher_mom = 0.995
        self.initial_lr = 5e-3
        self.num_epochs = 250
        self.mask_percentage = 0.75  # Default mask percentage for ConMAE
        self.config_plan.patch_size = (128, 128, 128)  # Patch size for ConMAE

    @override
    def build_architecture_and_adaptation_plan(
            self,
            config_plan,
            num_input_channels: int,
            num_output_channels: int,
            *args,
            **kwargs,
    ):
        """
        Builds the architecture and adaptation plan for the model.
        This method is overridden to provide specific architecture logic for Feature Contrastive Decoder.
        """
        # ---------------------------- Create architecture --------------------------- #
        architecture = FeatureContrastiveDecoderAligned(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            only_last_stage_as_latent=False,
            use_projector=True,
        )
        # --------------------- Build associated adaptation plan --------------------- #
        adapt_plan = self.save_adaption_plan(num_input_channels)
        return architecture, adapt_plan

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic for Feature Contrastive Decoder.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(
            device=self.device,
            recon_weight=5.0,
            fg_cos_weight=0.5,
            ntxent_weight=0.1,
            fine_grained_contrastive=False,
        )


class AlignedAETrainer(AlignedMAETrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the ConsisAETrainer with the given arguments.
        This class is specifically designed for training ConsisAE models.
        """
        super().__init__(*args, **kwargs)
        self.total_batch_size = 2
        self.config_plan.patch_size = (160, 160, 160)
        self.num_epochs = 1000
        self.initial_lr = 5e-3
        self.teacher_mom = 0.0
        self.mask_percentage = 0.00  # No masking for AutoEncoder

    def on_validation_epoch_start(self):
        # self.network.eval()

        # the predictor part of the model requires network to be in training mode...
        pass

    def shared_step(self, batch: dict, is_train: bool = True) -> dict:
        """
        Shared step for both training and validation.
        This method is overridden to provide specific shared step logic for ConsisAE.
        """
        data, bboxes = batch["all_crops"], batch["rel_bboxes"]

        data = data.to(self.device, non_blocking=True)
        bboxes = bboxes.to(self.device, non_blocking=True)

        with torch.no_grad():
            self.teacher.eval()
            teacher_output = self.teacher(data)
            # get the projections with stop gradient
            teacher_output = {
                k: v
                for k, v in teacher_output.items()
                if k == "proj" or k == "image_latent"
            }
            self.network.train()  # set the network to training mode

        mask = torch.zeros_like(data)

        if is_train:
            self.network.train()
            self.optimizer.zero_grad(set_to_none=True)

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            with torch.no_grad() if not is_train else dummy_context():
                output = self.network(data)
                loss_dict = self.loss(
                    model_output=output,
                    target=teacher_output,
                    gt_recon=data,
                    rel_bboxes=bboxes,
                    mask=mask,
                )
                l = loss_dict["loss"]

        if is_train:
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

            # update the teacher network with momentum of 0.95
            with torch.no_grad():
                self.ema(self.teacher, self.network, update_bn=False)

        return {k: v.detach().cpu().numpy() for k, v in loss_dict.items()}

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic for ConMAE.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(device=self.device, recon_weight=5.0, ntxent_weight=0.0)


class AlignedAEFTTrainer(AlignedAETrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the ConsisAETrainer with the given arguments.
        This class is specifically designed for training ConsisAE models.
        """
        super().__init__(*args, **kwargs)
        self.num_epochs = 250
        self.initial_lr = 5e-3
