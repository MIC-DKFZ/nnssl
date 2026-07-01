from dataclasses import dataclass, asdict, is_dataclass

import json

from nnssl.experiment_planning.experiment_planners.plan import Plan, ConfigurationPlan


def dataclass_to_dict(data):
    if is_dataclass(data):
        return {k: dataclass_to_dict(v) for k, v in asdict(data).items()}
    else:
        return data


@dataclass
class ConfigurationPlan_wandb(ConfigurationPlan):
    pass


@dataclass
class Plan_wandb(Plan):

    @staticmethod
    def load_from_file(path: str):
        json_dict: dict = json.load(open(path, "r"))
        configs = {
            k: ConfigurationPlan_wandb(**v)
            for k, v in json_dict["configurations"].items()
        }
        json_dict["configurations"] = configs
        return Plan(**json_dict)
