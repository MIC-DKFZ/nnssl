import argparse
from copy import deepcopy
import json

from nnssl.experiment_planning.experiment_planners.plan import Plan, ConfigurationPlan
import torch
from dataclasses import dataclass, fields
from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p
from torch.optim.optimizer import required
from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from os.path import join


def get_adaptation(
    arch: str,
    plan: dict,
    key_to_encoder: str,
    key_to_stem: str,
    keys_to_in_proj: tuple[str, ...],
    key_to_lpe: str | None = None,
    recommended_downstream_patchsize: tuple = [160, 160, 160],
):
    arch_plan = ArchitecturePlans(arch)
    adapt_plan = AdaptationPlan(
        architecture_plans=arch_plan,
        pretrain_plan=plan,
        pretrain_num_input_channels=1,
        recommended_downstream_patchsize=recommended_downstream_patchsize,
        key_to_encoder=key_to_encoder,
        key_to_stem=key_to_stem,
        keys_to_in_proj=keys_to_in_proj,
        key_to_lpe=key_to_lpe,
    )
    return adapt_plan


def add_pretrain_plan_entry():
    parser = argparse.ArgumentParser("Add adaptation infos to checkpoints")
    parser.add_argument("-i", type=str, help="path to checkpoint", required=True)
    parser.add_argument("-pl", type=str, help="path to plan", required=True)
    parser.add_argument("-o", type=str, help="output path", required=True)
    parser.add_argument("-arch", type=str, required=True)
    parser.add_argument("-key_to_encoder", type=str, required=True)
    parser.add_argument("-key_to_stem", type=str, required=True)
    parser.add_argument("-keys_to_in_proj", type=str, nargs="+", required=True)
    parser.add_argument("-key_to_lpe", type=str, default=None)
    parser.add_argument(
        "-recommended_downstream_patchsize",
        nargs="+",
        type=int,
        help="downstream patchsize recommendation. Default [160,160,160]",
        default=None,
    )
    parser.add_argument(
        "-spacing_style",
        type=str,
        help="Spacing style: [onemmiso, median, noresample], Default onemmiso",
        default="onemmiso",
    )

    args = parser.parse_args()
    if args.recommended_downstream_patchsize is None:
        recommended_downstream_patchsize = [160, 160, 160]
    else:
        recommended_downstream_patchsize = args.recommended_downstream_patchsize
    plan_dict = load_json(args.pl)
    plan_dict["configurations"]["onemmiso"] = plan_dict["configurations"].pop("3d_fullres")

    copy_dict = deepcopy(plan_dict)
    valid_keys_plan = {f.name for f in fields(Plan)}
    plan_dict = {k: v for k, v in copy_dict.items() if k in valid_keys_plan}

    # remove old keys from config and match configs
    valid_keys = {f.name for f in fields(ConfigurationPlan)}
    plan_dict["configurations"][args.spacing_style] = {
        k: v for k, v in copy_dict["configurations"][args.spacing_style].items() if k in valid_keys
    }
    plan_dict["configurations"][args.spacing_style]["spacing_style"] = args.spacing_style
    plan = Plan.from_dict(plan_dict)
    # print(plan)
    adapt_plan: AdaptationPlan = get_adaptation(
        args.arch,
        plan,
        key_to_encoder=args.key_to_encoder,
        key_to_stem=args.key_to_stem,
        keys_to_in_proj=tuple(args.keys_to_in_proj),
        key_to_lpe=args.key_to_lpe if "key_to_lpe" in args else None,
        recommended_downstream_patchsize=recommended_downstream_patchsize,
    )

    ckpt: dict = torch.load(args.i, weights_only=False)  # load the whole checkpoint

    # Drop unnecessary keys like optimizer, scheduler, etc.
    ckpt.pop("optimizer_state")
    ckpt.pop("grad_scaler_state")
    ckpt.pop("_best_ema")
    ckpt.pop("current_epoch")
    ckpt.pop("logging")
    if "spark_weights" in ckpt:
        ckpt.pop("spark_weights")
    # ckpt.pop("init_args")

    ckpt["nnssl_adaptation_plan"] = adapt_plan.serialize()
    checkpoint = AbstractBaseTrainer._convert_numpy(ckpt)
    # adapt_plan.pretrain_plan.configurations["onemmiso"].patch_size = [
    #     64,
    #     64,
    #     64,
    # ]  # Convert to allow weights_only=True

    AbstractBaseTrainer.verify_adaptation_plans(
        adapt_plan.serialize(), args.spacing_style, checkpoint["network_weights"]
    )
    output_dir = args.o
    maybe_mkdir_p(output_dir)
    torch.save(checkpoint, join(output_dir, "checkpoint_final.pth"))
    adapt_plan_serial = adapt_plan.serialize()
    adapt_plan_serial["trainer_name"] = checkpoint["trainer_name"]
    with open(join(output_dir, "adaptation_plan.json"), "w") as f:
        f.write(json.dumps(adapt_plan_serial, indent=4))
    # adapt_plan.serialize()
    print("Done saving checkpoint with adaptation plan to", output_dir)


if __name__ == "__main__":
    add_pretrain_plan_entry()
