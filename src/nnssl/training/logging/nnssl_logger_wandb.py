import os
from typing import Union
import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join

matplotlib.use("agg")
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
import wandb
from copy import deepcopy


class nnSSLLogger_wandb(object):

    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP
    """

    def __init__(self, verbose: bool = False, use_wandb: bool = False, wandb_init_args: dict = {},dataset_name: str = ""):
        self.my_fantastic_logging = {
            "train_losses": list(),
            "val_losses": list(),
            "lrs": list(),
            "epoch_start_timestamps": list(),
            "epoch_end_timestamps": list(),
        }
        self.verbose = verbose
        # shut up, this logging is great

        self.wandb = use_wandb
        if self.wandb:
            project_name = "nnssl_{}".format(dataset_name)
            run_id = os.getenv("WANDB_RUN_ID")
            maybe_resume_logging = self._maybe_resume_logging(wandb_init_args)
            if maybe_resume_logging:
                wandb.init(project=project_name, entity='mic_rocket', id=run_id, allow_val_change=True, resume=maybe_resume_logging, **wandb_init_args)
            else:
                wandb.init(project=project_name, entity='mic_rocket', id=run_id, allow_val_change=True, **wandb_init_args)

    def _maybe_resume_logging(self, wandb_init_args) -> Union[None, str]:
        """
        """
        # Check whether the env var WANDB_RUN_ID is set and if yes whether a logging folder already exists
        is_continuation = False
        if os.path.exists(os.path.join(wandb_init_args['dir'], 'wandb')):
            runs = [d for d in os.listdir(os.path.join(wandb_init_args['dir'], 'wandb'))]
            for run_dir in runs:
                if os.getenv("WANDB_RUN_ID") in run_dir:
                    os.environ["WANDB_RESUME"] = "must"
                    print(f"Found existing run {os.getenv('WANDB_RUN_ID')} in {run_dir}. Resuming logging.")
                    return "must"
            print(f"No existing run found in {wandb_init_args['dir']}. Starting new run.")
        return None
        

    def log(self, key, value, epoch: int):
        if self.wandb:
            if not len(self.my_fantastic_logging['val_losses'])==epoch:

                #if len(self.my_fantastic_logging['train_losses'])>0 and len(self.my_fantastic_logging['val_losses']):
                wandb.log({'train_loss': self.my_fantastic_logging['train_losses'][epoch],
                           'val_loss': self.my_fantastic_logging['val_losses'][epoch],
                           #'epoch_duration': self.my_fantastic_logging['epoch_end_timestamps'][epoch]-self.my_fantastic_logging['epoch_start_timestamps'][epoch],
                           'learning_rate': self.my_fantastic_logging['lrs'][epoch],
                           'epoch': epoch})
        """
        sometimes shit gets messed up. We try to catch that here
        """
        dict_content = self.my_fantastic_logging.get(key, None)
        if dict_content is None:
            raise ValueError("Trying to write unknown key to log dict")
        elif len(dict_content) != epoch:
            raise ValueError(f"Length of {key} list is '{len(dict_content)}'. Expected {epoch}")
        dict_content.append(value)
        return dict_content


    def wandb_log(self, key, value):
        wandb.log({key: value})

    def plot_progress_png(self, output_folder):
        # we infer the epoch form our internal logging
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1  # lists of epoch 0 have len 1
        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(3, 1, figsize=(30, 54))
        # regular progress.png as we are used to from previous nnU-Net versions
        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))
        ax.plot(
            x_values,
            self.my_fantastic_logging["train_losses"][: epoch + 1],
            color="b",
            ls="-",
            label="loss_tr",
            linewidth=4,
        )
        ax.plot(
            x_values,
            self.my_fantastic_logging["val_losses"][: epoch + 1],
            color="r",
            ls="-",
            label="loss_val",
            linewidth=4,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # epoch times to see whether the training speed is consistent (inconsistent means there are other jobs
        # clogging up the system)
        ax = ax_all[1]
        ax.plot(
            x_values,
            [
                i - j
                for i, j in zip(
                self.my_fantastic_logging["epoch_end_timestamps"][: epoch + 1],
                self.my_fantastic_logging["epoch_start_timestamps"],
            )
            ][: epoch + 1],
            color="b",
            ls="-",
            label="epoch duration",
            linewidth=4,
        )
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # learning rate
        ax = ax_all[2]
        ax.plot(
            x_values,
            self.my_fantastic_logging["lrs"][: epoch + 1],
            color="b",
            ls="-",
            label="learning rate",
            linewidth=4,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        plt.tight_layout()

        fig.savefig(join(output_folder, "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.my_fantastic_logging

    def load_checkpoint(self, checkpoint: dict):
        self.my_fantastic_logging = checkpoint
        # Check that all logs are of the same length. If not we concat to the shortest length and return this length
        max_length = max([len(i) for i in self.my_fantastic_logging.values()])
        min_length = min([len(i) for i in self.my_fantastic_logging.values()])
        assert max_length - min_length <= 1, "Lengths of logging items differ by more than 1. This is not supported."
        if max_length != min_length:
            logger.warning(
                f"WARNING: Lengths of logging items are not equal. Truncating all to the length of the shortest item ({min_length})"
                "This also sets the epoch number to the length to the minimum length -- Basically adding 1 epoch. This is a bit of a hack but it should work."
            )

        for key, value in self.my_fantastic_logging.items():
            self.my_fantastic_logging[key] = value[:min_length]
        return min_length

    def log_hypparams_to_wandb(self, trainer_class_instance_org, debug_dict):

        assert self.wandb, 'You need to use wandb for logging hyperparameters'
        trainer_class_instance = deepcopy(trainer_class_instance_org)

        for key, value in trainer_class_instance.my_init_kwargs['plan']['configurations']['3d_fullres'].items():
            if key in ['mask_ratio', 'vit_patch_size', 'embed_dim', 'encoder_eva_depth', 'encoder_eva_numheads', 'decoder_eva_depth', 'decoder_eva_numheads', 'initial_lr']:
                wandb.config.update({key: value}, allow_val_change=True)

        trainer_class_instance.my_init_kwargs.pop('pretrain_json')
        trainer_class_instance.my_init_kwargs.pop('plan')
        #import IPython
        #IPython.embed()

        #wandb.config.update(trainer_class_instance.my_init_kwargs)
        #wandb.config.update({k:v for (k,v) in trainer_class_instance.my_init_kwargs.items() if not callable(v)})
        for key, value in trainer_class_instance.my_init_kwargs.items():
            wandb.config.update({key: value}, allow_val_change=True)
        #debug_dict.pop("my_init_kwargs")
        #debug_dict.pop("configuration_manager")
        debug_dict.pop("device")
        debug_dict.pop("my_init_kwargs")
        debug_dict.pop("plan")
        debug_dict.pop("configuration_name")
        #debug_dict.pop("plan")
        #debug_dict.pop("dataset_json")
        #debug_dict.pop("unpack_dataset")
        debug_dict.pop("fold")
        debug_dict.pop("use_wandb")
        #wandb.config.update(debug_dict, allow_val_change=True)

