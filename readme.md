# nnSSL
This is the repository associated for the paper `Revisiting MAE pre-training for 3D medical image segmentation
` currently under review for ICLR 2025.
It contains the code for the experiments conducted in the paper and will be made public after the review process.
Currently it includes the following methods:
1. VoCo
2. VolumeFusion
3. Models Genesis
4. BaseMAE
5. Spark 3D

## Installation

1. Download the repository
2. Unzip and navigate into the repository
3. Install the repository `pip install .`

## Usage
To train new models with this repository you need to conduct the following steps.
(If you should have nnU-Net installed already you can set the `nnssl_raw` path to the same path as `nnUNet_raw_data_base`. It will allow you to use the data, just without the labels.)

### 1. Prepare the environment paths
To conduct pre-training this repo expects you to provide it with paths, similarly to nnU-Net these are:
- `nnssl_raw` - The path to the raw data
- `nnssl_preprocessed` - The path where the preprocessed raw data will be stored as preprocessed data
- `nnssl_results` - The path where results will be stored

### 2. Preprocess the data
Similarly to nnU-Net you need to preprocess data before training. To do so you need to:
1. Create a dataset in the [nnU-Net dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md). This means you need a `dataset.json` and a folder termed `imagesTr` which holds all the images. A `labelsTr` directory is not necessary (but if present it will be ignored). The `dataset.json` can be very simple, as it currently is only used as a remnant from nnU-Net. It will be removed in future versions. In the meanwhile the json provided below could be used if .
2. Run the preprocessing script `nnssl_plan_and_preprocess -d <Dataset ID>` with the respective dataset ID as an argument. The preprocessed data will be stored in the `nnssl_preprocessed` directory.

```json
// dataset.json
{
  "channel_names": {"0": "someMRI"},
  "description": "Unlabeled set of datapoints that are used for pre-text task pretraining",
  "file_ending": ".nii.gz",
  "licence": "Proprietary -- do not touch without permission",
  "name": "Some Images",
  "numTraining": 0,
  "release": "0.0",
}
```

### 3. Pretrain a model of choice

After preprocessing you can pre-train a model of choice. Simply choose `trainer` and `plan` and run the training script:
`python ./nnssl/run/run_training.py -tr SparkMAETrainer_BS6_1000ep -p nnsslPlans -num_gpus 1`

Some selected trainers that can be used are:
1. VoCo: `VoCoTrainer`
2. VolumeFusion: `VolumeFusionTrainer`
3. Models Genesis: `ModelGenesisTrainer`
4. Spark3D (fixed Masking): `SparkMAETrainer_BS6_1000ep`
5. Spark3D (var. Masking): `VariableSparkMAETrainer_BS6_ep1000`

Aside from these a substantial amount of other trainers are available in the `nnssl/training/nnSSLTrainer` directory and subdirectories.

### 4. Finetuning the model
After pre-training you can finetune the model on a downstream task. To do so you need to create a dataset in the nnU-Net format and follow our proposed fine-tuning scheme.

### Implementation of Spark3D
The implementation of Spark3D can be found in
`nnssl/architectures/spark_model`.