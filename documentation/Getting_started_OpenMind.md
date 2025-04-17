# Getting started - OpenMind meets SSL3D challenge

This is a guideline how to use the nnSSL framework with the OpenMind dataset. This is also the recommended starting point for the [SSL3D](https://ssl3d-challenge.dkfz.de/) challenge: 

## 1. Install nnssl
Follow the installation [instructions](/readme.md) and don't forget to set all necessary [env variables](/documentation/set_environment_variables.md). 

## 2. Download the dataset
You can find the OpenMind dataset on **[Hugging Face](https://huggingface.co/datasets/AnonRes/OpenMind)**. 
Follow the instructions of Hugging Face to download the data. 

## 3. Prepare the dataset
To prepare the dataset for pre-training you need to create a `pretrain_data.json`. The general structure is explained [here](/documentation/setting_up_paths.md)  
For the OpenNeuro Dataset we provide a [script](/src/nnssl/dataset_conversion/Dataset001_OpenMind.py) for conversion into the expected data format. 

## 4. Preprocess the dataset
You can preprocess the dataset by calling:

    1. nnssl_extract_fingerprint -d ID -np 20
    2. nnssl_plan_experiment -d ID
    3. nnssl_preprocess -d ID -np 12 -c CONFIG -part PARTID -total_parts MAXPARTS

- d points to the corresponding Dataset ID (745 for OpenNeuro)
- np specifies the number of worker
- c allows for defining the target spacing. We support the 1mm isotropic target spacin ('onemmiso'), median target spacing ('median'), and no fixed target spacing ('noresample').

In addition, you can distribute the preprocessing among multiple runs via: -part PARTID -total_parts MAXPARTS (If max parts is 5, partid should be between 0 and 4). 

## 5. Start a training
Now, it is getting exiting: To start a basic training for the ResencL and the Primus B architectures you can use the following commands: 

ResencL:

    nnssl_train ID CONFIG -tr BaseMAETrainer -p nnsslPlans 
    
Primus B:
    
    nnssl_train ID CONFIG -tr BaseEvaMAETrainer -p nnsslPlans

The ID corresponds to the dataset ID from above, and CONFIG corresponds to the defined target spacing ('onemmiso','median, 'noresample').
Here you can explore other implemented trainers, and you're also highly encouraged to implement your own SSL methods.
If you prefer not to use the 'nnssl' framework but still want to participate in the challenge, please refer to the 'build_architecture_and_adaptation_plan' function in the Trainer to learn how to load the network architecture.
To ensure your checkpoint is compatible with all downstream fine-tuning tasks, save the architecture following the example in the save_checkpoint function in [AbstractBaseTrainer](/src/nnssl/training/nnsslTrainer/AbstractTrainer.py).

For the SSL3D challenge, we will use the two network architectures from above and a fixed patchsize (160,160,160) for all downstream tasks.
An adaptation_plan.json file will be generated in the output folder, and its content will also be included in the final checkpoint. This file contains all the information needed for downstream fine-tuning. 

## 6. Downstream Usage with nnU-Net
We published a new fork to [nnU-Net](https://github.com/TaWald/nnUNet) for now - it will be merged to Main Repository in the future.
Follow the official nnU-NetV2 installation instructions. We expect the downstream dataset to be in nnU-Net format.

During the previous SSL training, all necessary information was saved directly in the checkpoint file. This includes both the preprocessing configuration and the network architecture, which are automatically transferred now for downstream usage.
To preprocess the downstream dataset, simply run:

    nnUNetv2_preprocess_like_nnssl -d ID -n NAME -pc PATH_TO_PRETRAINED_CKPT -am like_pretrained 

- n: Give the plan a name that allows you to identify the corresponding pre-training 
- am: Adaptation Mode - You can specify how to preprocess the downstream data given the pretrained config. Possible choices: fixed, default_nnunet, no_resample ,like_pretrained. If you select like pretrained, the target spacing will be the same as for the pre-training. 


Now you can train as you are used to:

ResEncL

    nnUNetv2_train_pretrained ID 3d_fullres FOLD -p NEWPLANSNAME

PrimusM

    nnUNetv2_train_pretrained ID 3d_fullres FOLD -p NEWPLANSNAME -tr PretrainedTrainer_Primus
    

The default trainer automatically detects the architecture and loads the corresponding weights. It applies a warm-up phase with an increasing learning rate for the first 50 epochs, followed by nnU-Net’s default polynomial learning rate decay schedule.
For the SSL3D Challenge, we will use shorter training schedules. The exact details will be shared after some initial testing.



