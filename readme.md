<!-- TODO: nnFoundation banner gif -->

## nnFoundation: 3D Foundation Models for Radiology
<sub>Copyright German Cancer Research Center (DKFZ) and contributors. Please make sure that your usage of this code is in compliance with its license.<sub>

This repository holds the code used to pre-train **nnFoundation**, our family of 3D radiology foundation models, built on top of the `nnssl` self-supervised learning framework.

## Model family

| Model | Architecture | Params | Trainer | Checkpoint |
|---|---|---|---|---|
| nnFoundationCNN | [ResEnc](https://arxiv.org/abs/2404.09556) | 102M | [`nnFoundationCNN_trainer`](src/nnssl/training/nnsslTrainer/nnFoundation/nnFoundationCNN.py#L109) | |
| nnFoundationViT-small | [Primus](https://arxiv.org/abs/2503.01835) | 205M | [`nnFoundationViT_small_trainer`](src/nnssl/training/nnsslTrainer/nnFoundation/nnFoundationViT.py#L120) | |
| nnFoundationViT-large | [Primus](https://arxiv.org/abs/2503.01835) | 674M | [`nnFoundationViT_large_trainer`](src/nnssl/training/nnsslTrainer/nnFoundation/nnFoundationViT.py#L124) | |

## Using nnFoundation

To fine-tune nnFoundation on your own downstream tasks, use one of our dedicated repositories:

- **Segmentation:** <!-- TODO: link -->
- **Detection:** <!-- TODO: link -->
- **Classification:** <!-- TODO: link -->
- **Report generation:** <!-- TODO: link -->

---

Below you will find a brief description of the needed steps, to get started with nnssl. 
Check-out the [documentation](documentation/) directory for a lot more information on how to use this repository.

## Installation

1. Download/clone the repository
2. Unzip and navigate into the repository
3. Install the repository `pip install -e .`  (-e optional)
4. Set the environment variables (see below)

More details can be found in the [installation instructions](documentation/installation_instructions.md).

<details>
<summary>Setting environment variables</summary>

In addition to the installation, this repository requires setting up three additional paths:

1. `nnssl_raw` -- The path holding datasets of raw `pretrain_data.json` files.
2. `nnssl_preprocessed` -- A path where preprocessed data will be stored.
3. `nnssl_results` -- A path where results will be stored.

More details can be found [here](documentation/setting_up_paths.md).

</details>

## Workflow
In order to conduct pre-training with this repository three main steps need to be conducted.

### 1. Raw Data Preparation
> This is a brief description. For a more detailed version, check [here](documentation/dataset_format.md). 

First, some pre-training dataset needs to be chosen. You can use the **[OpenMind dataset](https://huggingface.co/datasets/AnonRes/OpenMind)**. 
However any other dataset could be used as well. Opposed to nnU-Net, the data does not have to be in a specific format. Instead, a `pretrain_data.json` file needs to be created detailing the datasets specific information. (For simplicity the OpenMind dataset comes with this). To create this file for your own dataset or to understand the file, we refer to the instructions below.

> If you use the OpenMind dataset, use the dedicated call `nnssl_convert_openmind` to create its associated `pretrain_data.json`
<details>
<summary>Understanding and creating the `pretrain_data.json` file</summary>

Medical datasets generally center around studies of subjects. These subjects can be imaged in different sessions with different scanners or through different imaging protocols. This is reflected in the common BIDS data structure, which differentiates into:

  - `subjects` - The individual subjects in the dataset
  - `sessions` - The individual sessions of the subjects
  - `scans` - The individual scans of the sessions

  In our case, we are also interested in aggregating multiple datasets, hence we include

  - `dataset` - The individual datasets that was included

  All this information may be valuable for pre-training, e.g. one may want to develop a contrastive pre-training method that uses `scans` of the same `subject` during one `session` as positive pair and others as negative. Or one may want to develop a longitudinal `pre-training` that e.g. tries to predict the next scan of the next `session`. To allow using such information, we need to maintain this information in the `pretrain_data.json` file.
  <details>
  <summary>Hence, our `pretrain_data.json` file mirrors the BIDS structure:</summary>

  ```jsonc
  {  // Examplary Structure
    "collection_index": 745,    // Collection Index -- 
    "collection_name": "Dataset745__OpenNeuro",     // Collection Index -- 
    "datasets": {     // Dict of all datasets included
      "ds000001": {
        "dataset_index": "ds000001",
        "dataset_info": null,   // dict holding meta info of the dataset
        "subjects": {    // dict of all subjects
          "sub-01": {
            "subject_id": "sub-01",
            "subject_info": {
              "age": 26,
              "sex": "female",
            },
            "sessions": {     // Dict of all session and session information
              "ses-DEFAULT": {
                  "session_id": "ses-DEFAULT",
                  "session_info": null,
                  "images": [   // list of images -- Each image is it's own dictionary
                      {
                          "name": "sub-01_T1w.nii.gz",   // Image Name
                          "modality": "T1w",  // Modality of the Image
                          "image_info": {},  // Additional meta-data of the image (e.g. Scanner, etc.)
                          "image_path": "<Path_to_image>",
                          "associated_masks": {  // Associated masks of the image (if available)
                              "anatomy_mask": "<Path_to_associated_anatomy_mask>",
                              "anonymization_mask": "<Path_to_associated_anon_mask>",
                          },
                      },
                      ... // Additional images if taken in the session.
                  ],
              }
            }
          }
        }
      }
    }
  }
  ```
  </details>

  To generate this file, we recommend writing a python script that creates a `Collection` dataclass (located in `src/nnssl/data/raw_dataset.py`) and uses the `.to_dict()` method of the collection which will yield a valid `pretrain_data.json` file.

  To allow this file to be valid for differing machines, the file-paths support relative paths. 
  Relative paths are indicated through the pre-fix `$`. Moreover, when saving absolute paths the paths are checked, if the image path beginnings can be replaced by the paths in the **Environment Variables**: `["nnssl_raw", "nnssl_preprocessed"]`, replacing them with `$nnssl_raw` or `$nnssl_preprocessed` respectively.

</details>

### 2. Preprocessing the data
Currently the framework follows the nnU-Net preprocessing pipeline. This generally includes a *fingerprinting*, *planning*, and lastly *preprocessing* of the data. 
*Fingerprinting* determines overall shape and spacing of the data. 
*Planning* determines which patch size to use and which spacing to resample to.
*Preprocessing* normalizes, crops and resamples the data and saves it compressed in the `bloscv2` format[^1]. Moreover, the `pretrain_data.json` file will be copied to the `nnssl_preprocessed` directory, with the image and mask paths adjusted accordingly.


To conduct these three steps run

`nnssl_plan_and_preprocess -d <Dataset ID>`

[^1]: Bloscv2 is a compressed format that allows partial decompression and reading of the data, allowing fast I/O while minimizing CPU usage. 

### 3. Pretraining
Given the preprocessed data we can now pre-train the models. This is done by selecting a `trainer`a `dataset` and a `plan`. 
The `trainer` determines pre-training method and architecture, the `dataset` the data to use and the `plan` the preprocessing of the data.  

An exemplary pre-training call for a 4xGPU ResEnc-L MAE pre-training would be:
`python ./src/nnssl/run/run_training.py ID CONFIG -tr BaseMAETrainer_BS8 -p nnsslPlans -num_gpus 4`
or 
`nnssl_train ID CONFIG -tr BaseMAETrainer_BS8 -p nnsslPlans -num_gpus 4`

Note: Due to the lack of e.g. linear-probing for segmentation, no metrics aside from the train and validation loss are tracked during pre-training.

### 4. Adaptation
After pre-training, the resulting model checkpoint (or, pre-existing checkpoints) can be adapted to a specific downstream task.
This can be done via the dedicated [downstream repositories](#using-nnfoundation) linked above.


## Extending and Contributing
Due to the lack of established frameworks in the domain of 3D SSL, we are open to code contributions and extensions of the current framework.
