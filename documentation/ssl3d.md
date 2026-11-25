## SSL3D: Self-Supervised Learning for 3D Medical Imaging Challenge
<sub>Copyright German Cancer Research Center (DKFZ) and contributors. Please make sure that your usage of this code is in compliance with its license.<sub>

This page describes the public resources of the SSL3D challenge: the pretrained checkpoints of
all submissions, how to fine-tune them, and a runnable demo.

- [What SSL3D is](#what-ssl3d-is)
- [Checkpoints](#checkpoints)
- [Fine-tuning a checkpoint](#fine-tuning-a-checkpoint)
- [Demo: fine-tuning on MSD BrainTumour](#demo-fine-tuning-on-msd-braintumour)
- [Citation](#citation)

## What SSL3D is

The Self-Supervised Learning for 3D Medical Imaging (SSL3D) Challenge was held at MICCAI 2025.
It is a controlled benchmark of self-supervised pretraining for 3D medical imaging:

- All teams pretrained on the same dataset, [OpenMind](https://huggingface.co/datasets/MIC-DKFZ/OpenMind).
- There were two fixed architecture tracks: ResEnc-L (CNN) and Primus-M (Transformer).
- Teams submitted only their pretrained weights. Fine-tuning and evaluation on segmentation and
  classification tasks were done centrally by the organizers.

The pretraining framework of the challenge is this repository. Challenge design document:
[doi.org/10.5281/zenodo.15077452](https://doi.org/10.5281/zenodo.15077452)

## Checkpoints

All checkpoints are on Hugging Face:
[huggingface.co/MIC-DKFZ/SSL3D-checkpoints](https://huggingface.co/MIC-DKFZ/SSL3D-checkpoints)
(licence CC BY 4.0). All were pretrained on OpenMind. The files follow the layout

```
<track>/<team>/<team>_<model>.pth      # model = ResEncL or PrimusM
```

for example `ResEnc-L/DKFZ_MAE/DKFZ_MAE_ResEncL.pth`.

| Team (folder name) | Method | ResEnc-L | Primus-M |
|---|---|---|---|
| `DKFZ_MAE` | MAE (organizers' baseline) | ✓ | ✓ |
| `The_Latent_Campus` | Consistent View Alignment (CVA) | ✓ | ✓ |
| `Pep3Crew` | Anatomically grounded disentanglement | ✓ | – |
| `NTUST_EE304` | VoCo + MAE (ResEnc-L), MAE-filtered (Primus-M) | ✓ | ✓ |
| `HiLab` | Voxel-wise multi-scale self-distillation (Vox-MSD) | ✓ | ✓ |
| `PVG` | BigSimCLR-3D | ✓ | – |
| `gamma` | DINOv2 self-distillation | – | ✓ |
| `Datamen` | VoxCon | ✓ | ✓ |
| `Isikun.MMLab_and_C.H.Aarau.Radiology` | Variational Contrastive Learning (VCL) | ✓ | ✓ |
| `Anonymous` | Generalized Masked Autoencoder (GMAE) | ✓ | ✓ |

This gives 9 ResEnc-L and 8 Primus-M checkpoints. The code repository and paper of each team are
listed in the "Checkpoints" table of the
[model card](https://huggingface.co/MIC-DKFZ/SSL3D-checkpoints#checkpoints).

## Fine-tuning a checkpoint

### Segmentation (nnU-Net)

Segmentation fine-tuning uses the public [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). These are
the commands of the challenge protocol for ResEnc-L checkpoints, and the same commands the demo
runs:

```bash
nnUNetv2_extract_fingerprint -d <DATASET_ID>
nnUNetv2_plan_experiment -d <DATASET_ID>
nnUNetv2_preprocess_like_nnssl -d <DATASET_ID> -n SSL3D_ResEncL \
    -pc /path/to/DKFZ_MAE_ResEncL.pth -am like_pretrained
nnUNetv2_train_pretrained <DATASET_ID> 3d_fullres <FOLD> \
    -p ptPlans__SSL3D_ResEncL____<identifier> -tr PretrainedTrainer_150ep
```

`<identifier>` is printed by `nnUNetv2_preprocess_like_nnssl` (it holds the spacing and
normalisation, for example `Spacing__1.00_1.00_1.00___Norm__Z_Z_Z_Z` for a dataset with four
input channels).

Notes:

- **Download the checkpoint first.** nnU-Net can download a checkpoint from Hugging Face by
  itself (`-pc https://huggingface.co/...`), but only from a repository with
  `adaptation_plan.json` and `checkpoint_final.pth` in its root. The SSL3D repository has a
  different layout, so download the `.pth` file first and pass the local path, for example:
  ```python
  from huggingface_hub import hf_hub_download
  hf_hub_download(repo_id="MIC-DKFZ/SSL3D-checkpoints",
                  filename="ResEnc-L/DKFZ_MAE/DKFZ_MAE_ResEncL.pth", local_dir="pretrained_models")
  ```
- **Other submissions.** In the challenge, the plans were created once from the organizers' MAE
  checkpoint. Every other submission was then loaded into the same plans by adding
  `-overwrite_ckpt_path /path/to/<team>_ResEncL.pth` to `nnUNetv2_train_pretrained`. The demo
  does the same.
- **Baseline without pretraining.** Add `--from_scratch` to `nnUNetv2_train_pretrained`.
- **Primus-M.** In the challenge, Primus-M checkpoints were fine-tuned with AdamW and a learning
  rate of 1e-4. Follow the Primus instructions of the nnU-Net guide below for the trainer to use.
- Full guide, including other adaptation modes:
  [nnU-Net -- Fine-tuning from nnssl checkpoints](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/finetuning_from_nnssl_checkpoints.md)

### Classification

See [image_classification -- nnFoundation](https://github.com/MIC-DKFZ/image_classification/blob/master/docs/nnfoundation.md).

## Demo: fine-tuning on MSD BrainTumour

[`ssl3d/run_ssl3d_braintumour_demo.py`](ssl3d/run_ssl3d_braintumour_demo.py) fine-tunes a
pretrained SSL3D checkpoint on a public brain MRI segmentation dataset and compares it with the
same network trained from scratch, using only public resources:

- **Data:** Medical Segmentation Decathlon Task01 BrainTumour (multi-parametric MRI: FLAIR,
  T1w, T1gd, T2w; labels edema, non-enhancing tumour and enhancing tumour), public download
  without registration.
- **Model:** a ResEnc-L encoder submitted to the SSL3D challenge, pretrained on OpenMind and
  downloaded from
  [huggingface.co/MIC-DKFZ/SSL3D-checkpoints](https://huggingface.co/MIC-DKFZ/SSL3D-checkpoints).
  The default is the organizers' MAE baseline (`DKFZ_MAE`); any other ResEnc-L submission can
  be selected with `--team`.
- **Code:** the public nnU-Net repository ([github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)).
- **Protocol:** the segmentation fine-tuning protocol of the challenge. The data are resampled
  to the pretraining spacing (1 mm isotropic) and z-score normalised
  (`nnUNetv2_preprocess_like_nnssl -am like_pretrained`), and the network is trained with
  `PretrainedTrainer_150ep` (patch size 160 × 160 × 160, batch size 2, 150 epochs including
  15 warm-up epochs).
- **Comparison:** the same network, plans, and training schedule trained from scratch
  (`--from_scratch`), evaluated on a fixed 50/50 split (242 training / 242 test cases).

This dataset is not part of the challenge evaluation, so there are no reference values for it.

### 1. System requirements

#### Software and operating system

| Component | Requirement |
|---|---|
| Operating system | Linux. nnU-Net is tested on Ubuntu 18.04, 20.04 and 22.04, CentOS and RHEL |
| Python | 3.11 or newer |
| PyTorch | 2.1.2 or newer with CUDA support (nnU-Net excludes the 2.9 series) |
| nnU-Net | v2 from the public repository (version 2.9.0 at the time of writing); it installs its own dependencies |
| Other | `huggingface_hub` for the checkpoint download |

#### Versions the demo was tested on

| Item | Tested on |
|---|---|
| Operating system | Ubuntu 22.04.5 LTS (kernel 5.15) |
| Python | 3.11.18 |
| PyTorch | 2.15.0 (CUDA 13.2 build from PyPI, cuDNN 9.26) |
| NVIDIA driver | 580.178.04 |
| nnU-Net | 2.9.0, public master, commit 9920db74331c3a8a6813357c9ba4fde5e5104998 |

#### Hardware

- **Fine-tuning (this demo):** one NVIDIA GPU with at least 24 GB of memory (ResEnc-L, patch
  size 160 × 160 × 160, batch size 2). nnU-Net recommends at least 6 CPU cores for data
  augmentation.
- **Disk space:** about 40 GB: 14 GB for the MSD download (archive and extracted copy; the
  archive can be deleted after extraction), 7 GB for the nnU-Net raw data, 6 GB for the
  preprocessed data, 0.9 GB for the checkpoint, plus the training outputs and about 7 GB for
  the Python environment.
- **Pretraining (not part of this demo):** pretraining is done with this repository on
  OpenMind. The organizers' MAE baseline was pretrained on two A100 GPUs (80 GB); the hardware
  reported for the submitted methods ranged from one RTX 2080 Ti to twelve H100 GPUs.
- Internet access (GitHub, Hugging Face, AWS S3).

### 2. Installation guide

```bash
conda create -n ssl3d_demo python=3.11 -y
conda activate ssl3d_demo
```

The script installs PyTorch (default CUDA build from PyPI, only if PyTorch is missing),
nnU-Net from the public GitHub repository, and `huggingface_hub` into this environment.
If your system needs a specific PyTorch/CUDA build, install it first
(https://pytorch.org/get-started/locally/); the script then keeps it.
Use `--skip-install` if everything is already installed. The manual equivalent is:

```bash
pip install torch
pip install git+https://github.com/MIC-DKFZ/nnUNet.git huggingface_hub
```

The script itself needs only the Python standard library; installing `nnssl` is not required
for the demo.

**Typical install time:** about 5 to 10 minutes on a normal desktop computer, most of it the
PyTorch download.

### 3. Demo

#### Instructions

From the root of this repository:

```bash
python documentation/ssl3d/run_ssl3d_braintumour_demo.py --workdir /path/to/ssl3d_demo
```

This is the only path you need to set. The script sets all nnU-Net environment variables
for its own commands, inside the working directory:

| Variable | Location |
|---|---|
| `nnUNet_raw` | `<workdir>/nnUNet_raw` |
| `nnUNet_preprocessed` | `<workdir>/nnUNet_preprocessed` |
| `nnUNet_results` | `<workdir>/nnUNet_results/<team>` and `<workdir>/nnUNet_results/from_scratch` |

Any nnU-Net variables already set in your shell are overridden for the script's commands
only, so existing nnU-Net installations and data are not touched. The MSD download is
stored in `<workdir>/downloads`, the checkpoints in `<workdir>/pretrained_models`.

| Option | Effect |
|---|---|
| `--team NAME` | SSL3D ResEnc-L submission to fine-tune (default `DKFZ_MAE`); see the list below |
| `--skip-install` | do not (re)install nnU-Net |
| `--skip-scratch` | only fine-tune the pretrained model (halves the runtime, no comparison) |
| `--num-proc N` | number of processes for conversion and preprocessing (default 8) |
| `--no-compile` | disable `torch.compile` (slower training, but faster start) |
| `--nnunet-ref REF` | nnU-Net branch, tag, or commit to install (default `master`) |

Available teams: `DKFZ_MAE`, `The_Latent_Campus`, `Pep3Crew`, `NTUST_EE304`, `HiLab`, `PVG`,
`Datamen`, `Isikun.MMLab_and_C.H.Aarau.Radiology`, `Anonymous`.

The script can be re-run at any time: finished steps are skipped and an interrupted
training is resumed. Running it again with another `--team` reuses the data, the plans and
the from-scratch baseline.

#### What the script does

1. Installs PyTorch (if missing), nnU-Net (public repository), and `huggingface_hub`.
2. Downloads MSD Task01 BrainTumour (~7 GB) and converts it with `nnUNetv2_convert_MSD_dataset`
   to `Dataset001_BrainTumour` (the four MRI sequences become four input channels).
3. Creates the base plan (`nnUNetv2_extract_fingerprint`, `nnUNetv2_plan_experiment`).
4. Downloads `ResEnc-L/DKFZ_MAE/DKFZ_MAE_ResEncL.pth` from Hugging Face and runs
   `nnUNetv2_preprocess_like_nnssl`, which writes the plans
   `ptPlans__SSL3D_ResEncL____Spacing__1.00_1.00_1.00___Norm__Z_Z_Z_Z` and preprocesses the data.
   As in the challenge evaluation, these plans are created once from the organizers' MAE
   checkpoint; another submission is loaded into the same plans with `-overwrite_ckpt_path`.
5. Writes `splits_final.json` with the fixed train/test split (fold 0): cases sorted by name,
   every second case is a test case.
6. Fine-tunes the pretrained model, then trains the same configuration from scratch:
   ```bash
   nnUNetv2_train_pretrained 1 3d_fullres 0 -p <plans> -tr PretrainedTrainer_150ep -overwrite_ckpt_path <checkpoint.pth>
   nnUNetv2_train_pretrained 1 3d_fullres 0 -p <plans> -tr PretrainedTrainer_150ep --from_scratch
   ```
7. Evaluates both models on the 242 test cases and reports the paired difference.

#### Expected output

The demo produces the predicted segmentation masks of the 242 test cases, written as NIfTI
files to the results folder of each model
(`<workdir>/nnUNet_results/<team>/Dataset001_BrainTumour/<trainer>__<plans>__3d_fullres/fold_0/validation/`,
and under `nnUNet_results/from_scratch/` for the baseline), together with `summary.json`,
which holds the per-case metrics.

At the end, the script also prints the mean per-case Dice (averaged over the labels present
in each case) for both models and the paired difference with a bootstrap 95% confidence
interval:

```
=== Results on the held-out test cases
  DKFZ_MAE (pretrained)        mean per-case Dice 0.xxx on 242 cases   (nnU-Net foreground mean: 0.xxx)
  ResEnc-L (from scratch)      mean per-case Dice 0.xxx on 242 cases   (nnU-Net foreground mean: 0.xxx)

  Pretrained - scratch: +0.xxx Dice  (paired bootstrap 95% CI +0.xxx to +0.xxx; pretrained better in N/242 cases)
```

Small run-to-run differences are expected because training is not deterministic. The
training log in the `fold_0` folder reports that the pretrained weights were loaded.

#### Expected run time

The run time depends on the GPU. On an NVIDIA A100, one epoch takes around 100 s, so one
training of 150 epochs takes about 4 hours. The script runs two trainings (pretrained and
from scratch); download, preprocessing and the final prediction of the test cases come on
top. Training needs a GPU; the demo is not intended for a desktop computer without one.

Times of the steps before training (8 processes): download of MSD BrainTumour about
25 minutes (depends on the connection), conversion about 1 minute, fingerprint and planning
under 1 minute, preprocessing like the pretraining about 1 minute.

### 4. Instructions for use on your own data

Bring your dataset into the
[nnU-Net dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md),
set the three nnU-Net
[environment variables](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md),
download a checkpoint from the Hugging Face repository, and run the four commands of
[Fine-tuning a checkpoint](#segmentation-nnu-net).

Datasets with several input channels are supported: the pretrained input layer is repeated
across the channels. Add `--from_scratch` to the last command for the baseline without
pretraining. Classification fine-tuning and Primus-M checkpoints are covered by the guides
linked in that section.

### 5. Reproduction of the challenge results

The commands in [Fine-tuning a checkpoint](#segmentation-nnu-net) are the segmentation
fine-tuning protocol of the challenge, and all submitted checkpoints are public, so the
protocol can be applied to any dataset.

- **Public validation-stage datasets:** ISLES'22 and Yale-BrainMets (segmentation) and ABIDE
  (classification). Convert the segmentation datasets to the nnU-Net dataset format and run the
  commands above for each checkpoint; the challenge reports a 3-fold cross-validation on them.
- **Test-stage datasets:** seven private downstream datasets (four segmentation, three
  classification) that cannot be shared, so these results cannot be reproduced externally.

The challenge was run in 2025 with the framework versions current at that time. This repository
holds the maintained public versions, so small numerical differences to the reported results are
possible.

## Citation

SSL3D challenge paper: citation to be added on publication.

When using a checkpoint, please also cite OpenMind (see the [main readme](../readme.md#citation))
and the method paper of the team, where one exists; the list is in the
[model card](https://huggingface.co/MIC-DKFZ/SSL3D-checkpoints#citation).
