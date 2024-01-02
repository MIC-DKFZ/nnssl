from nnssl.architectures.get_network_from_plans import get_network_from_plans
from torch import nn

from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan


def build_network_architecture(
    config_plan: ConfigurationPlan,
    num_input_channels: int,
    num_output_channels: int,
) -> nn.Module:
    """
    This is where you build the architecture according to the plans. There is no obligation to use
    get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
    you want. Even ignore the plans and just return something static (as long as it can process the requested
    patch size)
    but don't bug us with your bugs arising from fiddling with this :-P
    This is the function that is called in inference as well! This is needed so that all network architecture
    variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
    training, so if you change the network architecture during training by deriving a new trainer class then
    inference will know about it).

    If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
    > label_manager = plans_manager.get_label_manager(dataset_json)
    > label_manager.num_segmentation_heads
    (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
    the number of outputs is != the number of classes. Also there is the ignore label for which no output
    should be generated. label_manager takes care of all that for you.)

    """
    return get_network_from_plans(
        configuration_plan=config_plan,
        num_input_channels=num_input_channels,
        num_output_channels=num_output_channels,
        deep_supervision=False,
    )
