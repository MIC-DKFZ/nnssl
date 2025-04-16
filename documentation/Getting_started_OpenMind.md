# Getting started - OpenMind meets SSL3D challenge

This is a guideline how to use the nnSSL framework with the OpenMind dataset. This is also the recommended starting point for the [SSL3D](https://ssl3d-challenge.dkfz.de/) challenge: 

## 1. Install nnssl
Follow the installation [instructions](/readme.md) and don#t forget to set all necessary env paths. 

## 2. Download the dataset
You can find the OpenMind dataset on **[Hugging Face](https://huggingface.co/datasets/AnonRes/OpenMind)**. 
Follow the instructions of Hugging Face to download the data. 

## 3. Prepare the dataset
To prepare the dataset for pre-training you need to create a `pretrain_data.json` file was explained [here](/readme.md)  
For the OpenNeuro Dataset we provide a [script](/src/nnssl/dataset_conversion/Dataset745_OpenMind.py) for conversion into the expected data format. 

## 4. Preprocess the dataset
You can preprocess the dataset by calling:

    1. nnssl_extract_fingerprint -d ID -np 20
    2. nnssl_plan_experiment -d ID
    3. nnssl_preprocess -d ID -np 12 -c CONFIG -part PARTID -total_parts MAXPARTS

-d points to the corresponding Dataset ID (745 for OpenNeuro)
-np specifies the number of worker
-c allows for defining the target spacing. We support the 1mm isotropic target spacin ('onemmiso'), median target spacing ('median'), and no fixed target spacing ('noresample').

In addition, you can distribute the preprocessing among multiple runs via: -part PARTID -total_parts MAXPARTS (If max parts is 5, partid should be between 0 and 4). 

## 5. Start a training
Now, it is getting exiting: To start a basic training for the ResencL and the Primus B architectures you can use the following commands: 

ResencL:

    nnssl_train ID CONFIG -tr BaseMAETrainer -p nnsslPlans 
    
Primus B:
    
    nnssl_train ID CONFIG -tr BaseEvaMAETrainer -p nnsslPlans

The ID corresponds to the dataset ID from above, and CONFIG corresponds to the defined target spacing ('onemmiso','median, 'noresample').
Here you can explore other implemented trainers, and you're also highly encouraged to implement your own SSL methods.
If you prefer not to use the 'nnssl' framework but still want to participate in the challenge, please refer to the 'build_architecture_and_adaptation_plan' function to access the network architecture.
To ensure your checkpoint is compatible with all downstream fine-tuning tasks, save the architecture following the example in the save_checkpoint function in 'AbstractBaseTrainer'. [AbstractBaseTrainer](/src/nnssl/training/nnsslTrainer/AbstractTrainer.py).

For the SSL3D challenge, we will use the two network architectures from above and a fixed patchsize (160,160,160) for all downstream tasks.
An adaptation_plan.json file will be generated in the output folder, and its content will also be included in the final checkpoint. This file contains all the information needed for downstream fine-tuning. 

## 6. Downstream Usage with nnU-Net






