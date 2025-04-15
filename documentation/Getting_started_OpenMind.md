# Getting started - OpenMind

This is a guideline how to use the nnSSL framework with the OpenMind dataset. This is also the recommended starting point for the SSL3D challenge: 

## 1. Install nnssl
Follow the installation [instructions](/readme.md) and don#t forget to set all necessary env paths. 

## 2. Download the dataset
You can find the OpenMind dataset on **[Hugging Face](https://huggingface.co/datasets/AnonRes/OpenMind)**. 
Follow the instructions of Hugging Face to download the data. 

## 3. Prepare the dataset
To prepare the dataset for pre-training you need to create a `pretrain_data.json` file was explained [here](/redme.md)  
For the OpenNeuro Dataset we provide a [script](/src/nnssl/dataset_conversion/Dataset745_OpenMind.py) for conversion into the expected data format. 

## 4. Preprocess the dataset
You can preprocess the dataset by calling:

    nnssl_preprocess -d ID -np 12 -c MODE

-d points to the corresponding Dataset ID (745 for OpenNeuro)
-np specifies the number of worker
-c allows for defining the target spacing. We support the 1mm isotropic target spacin ('onemmiso'), median target spacing ('median'), and no fixed target spacing ('noresample').

In addition, you can distribute the preprocessing among multiple runs via: -part PARTID -total_parts MAXPARTS (If max parts is 5, partid should be between 0 and 4). 

## Start a training





