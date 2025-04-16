import argparse

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p
from torch.optim.optimizer import required
from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans


adaptation_dict = {"ResEncL": {'key_to_encoder':"encoder.stages",
                               'key_to_stem' : "encoder.stem",
                               'keys_to_in_proj' : ("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0"),
                               'key_to_lpe' : None},
                   "PrimusM": {'key_to_encoder' : "eva",
                               'key_to_stem' : "down_projection",
                               'keys_to_in_proj' : ("down_projection.proj",),
                               'key_to_lpe' : "eva.pos_embed",}}

def get_adaptation(arch:str, plan:dict, recommended_downstream_patchsize : tuple=[160,160,160]):
    assert arch in adaptation_dict.keys(), "Architecture must be in ResEncL or PrimusL."
    arch_plan = ArchitecturePlans(arch)
    adapt_plan = AdaptationPlan(
        architecture_plans=arch_plan,
        pretrain_plan=plan,
        pretrain_num_input_channels=1,
        recommended_downstream_patchsize=recommended_downstream_patchsize,
        key_to_encoder=adaptation_dict[arch]["key_to_encoder"],
        key_to_stem=adaptation_dict[arch]["key_to_stem"],
        keys_to_in_proj=adaptation_dict[arch]["keys_to_in_proj"]  ,
        key_to_lpe=adaptation_dict[arch]["key_to_lpe"],
    )
    return adapt_plan


def add_pretrain_plan_entry():
    parser = argparse.ArgumentParser(
        "Add adaptation infos to checkpoints"
    )
    parser.add_argument(
        "-i",
        type=str,
        help="path to checkpoint",
        required=True)
    parser.add_argument(
        "-pl",
        type=str,
        help="path to plan",
        required=True)
    parser.add_argument(
        "-o",
        type=str,
        help="output path",
        required=True)
    parser.add_argument(
        "-arch",
        type=str,
        required=True)
    parser.add_argument(
        "-recommended_downstream_patchsize",
        nargs="+",
        type=int,
        help="downstream patchsize recommendation. Default [160,160,160]",
        default=None,
    )

    args = parser.parse_args()
    if args.recommended_downstream_patchsize is None:
        recommended_downstream_patchsize = [160,160,160]
    else:
        recommended_downstream_patchsize = args.recommended_downstream_patchsize

    plan = load_json(args.pl)
    adapt_plan = get_adaptation(args.arch, plan, recommended_downstream_patchsize)

    ckpt = torch.load(args.i)

    ckpt["nnssl_adaptation_plan"] = adapt_plan.serialize()
    checkpoint = AbstractBaseTrainer._convert_numpy(ckpt)
    torch.save(checkpoint, args.o)



if __name__ == "__main__":
    add_pretrain_plan_entry()
