import gc
import copy
from typing import Union, Tuple, List

import torch
import numpy as np

from torch import autocast
from typing_extensions import override

from nnssl.architectures.consis_arch import (
    ConsisEvaMAE,
    FeatureContrastiveDecoderAlignedEva,
)
from nnssl.utilities.helpers import dummy_context
from nnssl.ssl_data.dataloading.aligned_transform import OverlapTransform
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseEvaMAETrainer import (
    BaseEvaMAETrainer,
)
from nnssl.ssl_data.configure_basic_dummyDA import (
    configure_rotation_dummyDA_mirroring_and_inital_patch_size,
)


class BaseAlignedMAETrainer(BaseEvaMAETrainer):
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
        self.total_batch_size = 4
        self.initial_lr = 1e-4  # Initial learning rate for the optimizer
        self.num_epochs = 250
        self.warmup_duration_whole_net = 10  # Warmup duration for the whole network
        self.teacher = None
        self.teacher_mom = 0.995  # Momentum for the teacher model
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
            patch_size=self.config_plan.patch_size,
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

    @override
    def build_architecture_and_adaptation_plan(
        self, config_plan, num_input_channels, num_output_channels
    ):
        network = ConsisEvaMAE(
            input_channels=num_input_channels,
            embed_dim=self.embed_dim,
            patch_embed_size=self.vit_patch_size,
            output_channels=num_output_channels,
            input_shape=tuple(self.config_plan.patch_size),
            encoder_eva_depth=self.encoder_eva_depth,
            encoder_eva_numheads=self.encoder_eva_numheads,
            decoder_eva_depth=self.decoder_eva_depth,
            decoder_eva_numheads=self.decoder_eva_numheads,
            patch_drop_rate=self.mask_percentage,
            drop_path_rate=self.drop_path_rate,
            attn_drop_rate=self.attention_drop_rate,
            init_values=self.init_value,
            scale_attn_inner=self.scale_attn_inner,
        )
        adapt_plan = self.save_adaption_plan(num_input_channels)
        return network, adapt_plan

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
            # If momentum is 0, we just copy the student model to the teacher model
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
            self.teacher.eval()  # ensure the teacher is in eval mode
            teacher_output = self.teacher(data)
            # del all keys that are not `proj`
            teacher_output = {
                k: v
                for k, v in teacher_output.items()
                if k == "proj" or k == "image_latent"
            }
            self.network.train()

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
                # Forward pass with PatchDropout
                output = self.network(data)
                mask = self.create_mask(
                    output["keep_indices"],
                    self.config_plan.patch_size,
                    self.vit_patch_size,
                )

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
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.grad_clip
                )
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                l.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.grad_clip
                )
                self.optimizer.step()

            with torch.no_grad():
                self.ema(self.teacher, self.network, update_bn=False)

        del data, bboxes, output, teacher_output, mask
        gc.collect()

        return {k: v.detach().cpu().numpy() for k, v in loss_dict.items()}

    def train_step(self, batch: dict) -> dict:
        """
        Training step for the model.
        This method is overridden to provide specific training logic.
        """
        return self.shared_step(batch, is_train=True)

    def validation_step(self, batch: dict) -> dict:
        """
        Validation step for the model.
        This method is overridden to provide specific validation logic.
        """
        return self.shared_step(batch, is_train=False)

################################### We start configuration basic here #####################
class AlignedMAEEvaTrainer(BaseAlignedMAETrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the ConsisMAEEvaTrainer with the given arguments.
        This class is specifically designed for training ConsisMAE models.
        """
        super().__init__(*args, **kwargs)
        self.teacher_mom = 0.995
        self.total_batch_size = 4
        self.initial_lr = 3e-4  # Initial learning rate for the optimizer
        self.num_epochs = 500
        self.warmup_duration_whole_net = 10  # Warmup duration for the whole network

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        # Create the loss function
        return AlignedMAELoss(device=self.device, recon_weight=5.0)

################################### We start configuration basic here  step1 #####################

class AlignedMAEFTLR3EvaTrainer(AlignedMAEEvaTrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the ConsisMAEFTEvaTrainer with the given arguments.
        """
        super().__init__(*args, **kwargs)
        self.initial_lr = 3e-4
        self.num_epochs = 150

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(
            device=self.device, recon_weight=5.0, fg_cos_weight=1.0, ntxent_weight=0.1
        )


class AlignedMAEFTNoConEvaLR3Trainer(AlignedMAEFTLR3EvaTrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the FeatConDecAlignedMAEFTEvaTrainer with the given arguments.
        This class is specifically designed for training models with feature contrastive loss.
        """
        super().__init__(*args, **kwargs)
        self.teacher_mom = 0.995
        self.total_batch_size = 4
        self.initial_lr = 3e-4  # Initial learning rate for the optimizer
        self.num_epochs = 50
        self.mask_percentage = (
            0.75  # Mask percentage for the feature contrastive decoder
        )
        self.warmup_duration_whole_net = 5  # Warmup duration for the whole network

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(
            device=self.device, recon_weight=5.0, fg_cos_weight=1.0, ntxent_weight=0.0
        )

################################### We start configuration basic here  step2 #####################

class GramAlignedMAEFTNoConEvaLR3Trainer(AlignedMAEFTLR3EvaTrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the FeatConDecAlignedMAEFTEvaTrainer with the given arguments.
        This class is specifically designed for training models with feature contrastive loss.
        """
        super().__init__(*args, **kwargs)
        self.teacher_mom = 0.995
        self.total_batch_size = 4
        self.initial_lr = 3e-4  # Initial learning rate for the optimizer
        self.num_epochs = 50
        self.mask_percentage = (
            0.75  # Mask percentage for the feature contrastive decoder
        )
        self.warmup_duration_whole_net = 5  # Warmup duration for the whole network

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic.
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

################################### We start configuration basic here  step3 #####################

class GramAlignedMAEFTNoConNoProjEvaLR3Trainer(GramAlignedMAEFTNoConEvaLR3Trainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the FeatConDecAlignedMAEFTEvaTrainer with the given arguments.
        This class is specifically designed for training models with feature contrastive loss.
        """
        super().__init__(*args, **kwargs)
        self.teacher_mom = 0.995
        self.total_batch_size = 4
        self.initial_lr = 3e-4  # Initial learning rate for the optimizer
        self.num_epochs = 50
        self.mask_percentage = (
            0.75  # Mask percentage for the feature contrastive decoder
        )
        self.warmup_duration_whole_net = 5  # Warmup duration for the whole network

    def build_architecture_and_adaptation_plan(
        self, config_plan, num_input_channels, num_output_channels
    ):
        network = ConsisEvaMAE(
            input_channels=num_input_channels,
            embed_dim=self.embed_dim,
            patch_embed_size=self.vit_patch_size,
            output_channels=num_output_channels,
            input_shape=tuple(self.config_plan.patch_size),
            encoder_eva_depth=self.encoder_eva_depth,
            encoder_eva_numheads=self.encoder_eva_numheads,
            decoder_eva_depth=self.decoder_eva_depth,
            decoder_eva_numheads=self.decoder_eva_numheads,
            patch_drop_rate=self.mask_percentage,
            drop_path_rate=self.drop_path_rate,
            attn_drop_rate=self.attention_drop_rate,
            init_values=self.init_value,
            scale_attn_inner=self.scale_attn_inner,
            use_projector=False,
        )
        adapt_plan = self.save_adaption_plan(num_input_channels)
        return network, adapt_plan


class GramAlignedMAEFTConEvaLR3Trainer(AlignedMAEFTLR3EvaTrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the FeatConDecAlignedMAEFTEvaTrainer with the given arguments.
        This class is specifically designed for training models with feature contrastive loss.
        """
        super().__init__(*args, **kwargs)
        self.teacher_mom = 0.995
        self.total_batch_size = 4
        self.initial_lr = 3e-4  # Initial learning rate for the optimizer
        self.num_epochs = 50
        self.mask_percentage = (
            0.75  # Mask percentage for the feature contrastive decoder
        )
        self.warmup_duration_whole_net = 5  # Warmup duration for the whole network

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic.
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


class GramAlignedMAEFTConNoProjEvaLR3Trainer(GramAlignedMAEFTConEvaLR3Trainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the FeatConDecAlignedMAEFTEvaTrainer with the given arguments.
        This class is specifically designed for training models with feature contrastive loss.
        """
        super().__init__(*args, **kwargs)
        self.teacher_mom = 0.995
        self.total_batch_size = 4
        self.initial_lr = 3e-4  # Initial learning rate for the optimizer
        self.num_epochs = 50
        self.mask_percentage = (
            0.75  # Mask percentage for the feature contrastive decoder
        )
        self.warmup_duration_whole_net = 5  # Warmup duration for the whole network

    def build_architecture_and_adaptation_plan(
        self, config_plan, num_input_channels, num_output_channels
    ):
        network = ConsisEvaMAE(
            input_channels=num_input_channels,
            embed_dim=self.embed_dim,
            patch_embed_size=self.vit_patch_size,
            output_channels=num_output_channels,
            input_shape=tuple(self.config_plan.patch_size),
            encoder_eva_depth=self.encoder_eva_depth,
            encoder_eva_numheads=self.encoder_eva_numheads,
            decoder_eva_depth=self.decoder_eva_depth,
            decoder_eva_numheads=self.decoder_eva_numheads,
            patch_drop_rate=self.mask_percentage,
            drop_path_rate=self.drop_path_rate,
            attn_drop_rate=self.attention_drop_rate,
            init_values=self.init_value,
            scale_attn_inner=self.scale_attn_inner,
            use_projector=False,
            use_projector_global=True
        )
        adapt_plan = self.save_adaption_plan(num_input_channels)
        return network, adapt_plan


class ConMAEFTEvaLR3Trainer(AlignedMAEFTLR3EvaTrainer):

    def build_loss(self):
        """
        Builds the loss function for the model.
        This method is overridden to provide specific loss logic.
        """
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(
            device=self.device, recon_weight=5.0, fg_cos_weight=0.0, ntxent_weight=0.1
        )


class AlignedConConMAEFTEvaTrainer(AlignedMAEFTLR3EvaTrainer):

    def build_loss(self):
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(
            device=self.device,
            recon_weight=5.0,
            fg_cos_weight=0.2,
            ntxent_weight=0.1,
            fine_grained_contrastive=True,
            out_size=5,
        )


class FeatConDecAlignedMAEFTEvaTrainer(AlignedMAEFTLR3EvaTrainer):

    def __init__(self, *args, **kwargs):
        """
        Initialize the FeatConDecAlignedMAEFTEvaTrainer with the given arguments.
        This class is specifically designed for training models with feature contrastive loss.
        """
        super().__init__(*args, **kwargs)
        self.teacher_mom = 0.995
        self.total_batch_size = 4
        self.initial_lr = 3e-4  # Initial learning rate for the optimizer
        self.num_epochs = 50
        self.mask_percentage = (
            0.75  # Mask percentage for the feature contrastive decoder
        )
        self.warmup_duration_whole_net = 5  # Warmup duration for the whole network

    @override
    def build_architecture_and_adaptation_plan(
        self, config_plan, num_input_channels, num_output_channels
    ):
        network = FeatureContrastiveDecoderAlignedEva(
            input_channels=num_input_channels,
            embed_dim=self.embed_dim,
            patch_embed_size=self.vit_patch_size,
            output_channels=num_output_channels,
            input_shape=tuple(self.config_plan.patch_size),
            encoder_eva_depth=self.encoder_eva_depth,
            encoder_eva_numheads=self.encoder_eva_numheads,
            decoder_eva_depth=self.decoder_eva_depth,
            decoder_eva_numheads=self.decoder_eva_numheads,
            patch_drop_rate=self.mask_percentage,
            drop_path_rate=self.drop_path_rate,
            attn_drop_rate=self.attention_drop_rate,
            init_values=self.init_value,
            scale_attn_inner=self.scale_attn_inner,
        )
        adapt_plan = self.save_adaption_plan(num_input_channels)
        return network, adapt_plan

    def build_loss(self):
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(
            device=self.device,
            recon_weight=5.0,
            fg_cos_weight=0.5,
            ntxent_weight=0.1,
            fine_grained_contrastive=False,
            out_size=5,
        )


class AlignedAEEvaTrainer(AlignedMAEEvaTrainer):
    """
    Trainer for ConsisAE with a mask percentage of 10%.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the ConsisAEEvaTrainer with the given arguments.
        """
        super().__init__(*args, **kwargs)
        self.teacher_mom = 0.0
        self.mask_percentage = 0.0
        self.total_batch_size = (
            2  # since we don't mask anything, we need to reduce the batch size
        )
        self.mask_percentage = 0.0  # No masking for AlignedAE

    def build_loss(self):
        from nnssl.training.loss.aligned_mae_loss import AlignedMAELoss

        return AlignedMAELoss(device=self.device, recon_weight=5.0, ntxent_weight=0.0)

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
            # ensure the teacher is in eval mode
            teacher_output = self.teacher(data)
            # del all keys that are not `proj`
            teacher_output = {
                k: v
                for k, v in teacher_output.items()
                if k == "proj" or k == "image_latent"
            }
            self.network.train()

        if is_train:
            self.network.train()
            self.optimizer.zero_grad(set_to_none=True)

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            with torch.no_grad() if not is_train else dummy_context():
                # Forward pass with PatchDropout
                output = self.network(data)
                mask = torch.zeros_like(output["recon"])  # pretend everything is masked

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
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.grad_clip
                )
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                l.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.grad_clip
                )
                self.optimizer.step()

            with torch.no_grad():
                self.ema(self.teacher, self.network, update_bn=False)

        return {k: v.detach().cpu().numpy() for k, v in loss_dict.items()}


class AlignedAEFTEvaTrainer(AlignedAEEvaTrainer):
    """
    Trainer for ConsisAE with a mask percentage of 10% and a learning rate of 3e-4.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the AlignedAEFTEvaTrainer with the given arguments.
        """
        super().__init__(*args, **kwargs)
        self.initial_lr = 3e-4
        self.num_epochs = 150
