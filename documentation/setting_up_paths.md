# Setting up Paths

nnssl relies on environment variables to know where your dataset defition json files, preprocessed data and trained model weights are stored. 
To use the full functionality of nnssl, the following three environment variables must be set:

1) `nnssl_raw`: This is where you place your `pretrain_data.json` files. Similarly to nnU-Net each folder  
DatasetXXX_YYY where XXX is a 3-digit identifier (such as 001, 002, 043, 999, ...) and YYY is the (unique) 
dataset name. To use a dataset it must contain a `pretrain_data.json` file which contains paths to the actual data.
This allows you to easily create combinations of your datasets you have lieing around.
To create these `pretrain_data.json` files please follow the [dataset format instructions](dataset_format.md).

    Example structure:
    ```
    nnssl_raw/Dataset001_NAME1
    └── pretrain_data.json
    nnUNet_raw/Dataset002_NAME2
    └── pretrain_data.json
    ...
    ```

2) `nnssl_preprocessed`: Once you choose to preprocess data it will be pulled from wherever it is stored (as specified in the `pretrain_data.json`) and written to this directory.
The data will also be read from this folder during training. It is important that this folder is located on a drive with low access latency and high 
throughput (such as a nvme SSD (PCIe gen 3 is sufficient)).

3) `nnssl_results`: This specifies where nnssl will save the model weights.

### How to set environment variables
See [here](set_environment_variables.md).