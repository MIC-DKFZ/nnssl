#!/usr/bin/env python3
"""
SSL3D demo: fine-tune a pretrained SSL3D challenge checkpoint on a public brain MRI dataset
and compare it against the identical network trained from scratch.

Dataset : Medical Segmentation Decathlon Task01 BrainTumour (multi-parametric MRI: FLAIR, T1w,
          T1gd, T2w; labels edema, non-enhancing tumour, enhancing tumour), public, no
          registration (CC-BY-SA 4.0).
Split   : fixed 50/50 train/test split: cases sorted by name, every second case is a test case
          (242 training / 242 test cases for the 484 labelled cases).
Model   : a ResEnc-L encoder submitted to the SSL3D challenge (default: the organizers' MAE
          baseline), pretrained on OpenMind and downloaded from Hugging Face
          (https://huggingface.co/MIC-DKFZ/SSL3D-checkpoints).
Protocol: the segmentation fine-tuning protocol of the challenge: data resampled to the
          pretraining spacing (1 mm isotropic), z-score normalisation, the patch size recommended
          by the checkpoint (160^3), batch size 2, 150 epochs including 15 warm-up epochs, SGD
          (nnUNetv2_preprocess_like_nnssl -am like_pretrained + PretrainedTrainer_150ep).
Baseline: same plans / trainer / schedule, trained with --from_scratch.

Everything uses the public nnU-Net repository (https://github.com/MIC-DKFZ/nnUNet).

Requirements
  - Linux, one NVIDIA GPU with >= 24 GB memory (ResEnc-L, patch 160^3, batch size 2), CUDA-enabled PyTorch
  - Python >= 3.11, ~40 GB free disk space (download, raw + preprocessed data, checkpoint, results, environment)
  - Runtime: depends on the GPU; around 100 s per epoch on an A100, i.e. about 4 hours per
    training of 150 epochs (two trainings: pretrained and from scratch)

Usage
  # in a fresh Python >= 3.11 environment (PyTorch and nnU-Net are installed by the script):
  python run_ssl3d_braintumour_demo.py --workdir /path/to/ssl3d_demo

  Options: --team <name> (which SSL3D submission to fine-tune, default DKFZ_MAE),
           --skip-install, --skip-scratch (only fine-tune the pretrained model),
           --num-proc N (preprocessing workers), --no-compile,
           --nnunet-ref <branch|tag|commit>

All steps are idempotent: re-running the script skips finished steps and resumes training.
"""
import argparse
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

# ----------------------------------------------------------------------------- configuration
HF_REPO = "MIC-DKFZ/SSL3D-checkpoints"
NNUNET_REPO = "https://github.com/MIC-DKFZ/nnUNet.git"
MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
DATASET_ID = 1
DATASET_NAME = "Dataset001_BrainTumour"
PRETRAIN_NAME = "SSL3D_ResEncL"
TRAINER = "PretrainedTrainer_150ep"
FOLD = "0"

# ResEnc-L submissions in the Hugging Face repository: <track>/<team>/<team>_ResEncL.pth
TEAMS = ["DKFZ_MAE", "The_Latent_Campus", "Pep3Crew", "NTUST_EE304", "HiLab", "PVG", "Datamen",
         "Isikun.MMLab_and_C.H.Aarau.Radiology", "Anonymous"]
# As in the challenge evaluation, the fine-tuning plans are created once from the organizers' MAE
# checkpoint; any other submission is then loaded into the same plans with -overwrite_ckpt_path.
PLAN_TEAM = "DKFZ_MAE"


# ----------------------------------------------------------------------------- helpers
def log(msg):
    print(f"\n=== {msg}", flush=True)


def run(cmd, env):
    cmd = [nnunet_bin(cmd[0])] + cmd[1:]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def install_dependencies(ref):
    """Install PyTorch (if missing), the public nnU-Net and huggingface_hub into this Python environment."""
    try:
        import torch  # noqa: F401
    except ImportError:
        log("Installing PyTorch (default CUDA build from PyPI; install a specific build beforehand if needed)")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch"], check=True)
    log(f"Installing public nnU-Net ({NNUNET_REPO}@{ref}) and huggingface_hub")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade",
                    f"git+{NNUNET_REPO}@{ref}", "huggingface_hub"], check=True)
    out = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__, torch.cuda.device_count())"],
                         check=True, capture_output=True, text=True).stdout.split()
    print(f"PyTorch {out[0]}, {out[1]} CUDA device(s) visible")
    if out[1] == "0":
        sys.exit("No CUDA GPU visible to PyTorch. Training requires an NVIDIA GPU.")


def nnunet_bin(name):
    """nnU-Net console scripts live next to the Python interpreter running this script."""
    p = Path(sys.executable).parent / name
    return str(p) if p.exists() else name


def setup_env(workdir: Path):
    env = os.environ.copy()
    for var, sub in [("nnUNet_raw", "nnUNet_raw"), ("nnUNet_preprocessed", "nnUNet_preprocessed")]:
        p = workdir / sub
        p.mkdir(parents=True, exist_ok=True)
        env[var] = str(p)
    return env


def results_env(env, workdir: Path, name: str):
    """Each model gets its own nnUNet_results folder, so several submissions can share one workdir."""
    p = workdir / "nnUNet_results" / name
    p.mkdir(parents=True, exist_ok=True)
    return dict(env, nnUNet_results=str(p))


def download_msd(workdir: Path):
    tar_path = workdir / "downloads" / "Task01_BrainTumour.tar"
    out_dir = workdir / "downloads" / "Task01_BrainTumour"
    if (out_dir / "dataset.json").exists():
        return out_dir
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    if not tar_path.exists():
        log(f"Downloading MSD Task01 BrainTumour (~7 GB) from {MSD_URL}")
        tmp = tar_path.with_suffix(".part")

        def hook(blocks, bs, total):
            if total > 0 and blocks % 2000 == 0:
                print(f"  {min(100, blocks * bs * 100 / total):5.1f}%", flush=True)
        urllib.request.urlretrieve(MSD_URL, tmp, hook)
        tmp.rename(tar_path)
    log("Extracting")
    with tarfile.open(tar_path) as tf:
        tf.extractall(tar_path.parent)
    return out_dir


def download_checkpoint(workdir: Path, team: str):
    """Download <track>/<team>/<team>_ResEncL.pth from the SSL3D Hugging Face repository."""
    rel = f"ResEnc-L/{team}/{team}_ResEncL.pth"
    local_dir = workdir / "pretrained_models"
    target = local_dir / rel
    if not target.exists():
        log(f"Downloading {rel} from https://huggingface.co/{HF_REPO}")
        code = ("from huggingface_hub import hf_hub_download; "
                f"hf_hub_download(repo_id={HF_REPO!r}, filename={rel!r}, local_dir={str(local_dir)!r})")
        subprocess.run([sys.executable, "-c", code], check=True)
    assert target.exists(), f"download failed: {target}"
    return target


def find_plans(pp: Path):
    """nnUNetv2_preprocess_like_nnssl names the plans ptPlans__<name>____<spacing and normalisation>."""
    cands = sorted(pp.glob(f"ptPlans__{PRETRAIN_NAME}____*.json"))
    return cands[0].stem if cands else None


def write_split(env):
    """Fixed 50/50 split: cases sorted by name, every second case is a test case (fold 0 validation set)."""
    pp = Path(env["nnUNet_preprocessed"]) / DATASET_NAME
    raw_imgs = Path(env["nnUNet_raw"]) / DATASET_NAME / "imagesTr"
    all_cases = sorted({f.name[:-len("_0000.nii.gz")] for f in raw_imgs.glob("*_0000.nii.gz")})
    assert len(all_cases) > 1, f"no cases found in {raw_imgs}"
    test, train = all_cases[1::2], all_cases[0::2]
    pp.mkdir(parents=True, exist_ok=True)
    with open(pp / "splits_final.json", "w") as f:
        json.dump([{"train": train, "val": test}], f, indent=1)
    print(f"splits_final.json: fold 0 = {len(train)} train / {len(test)} test cases (fixed split)")


def find_result_dir(env, scratch: bool):
    res = Path(env["nnUNet_results"]) / DATASET_NAME
    cands = [d for d in res.glob(f"{TRAINER}__ptPlans__{PRETRAIN_NAME}*__3d_fullres")
             if ("from_scratch" in d.name) == scratch]
    return cands[0] if cands else None


def per_case_dice(summary_file):
    """Per-case Dice averaged over the labels present in the reference (edema, non-enhancing, enhancing tumour)."""
    s = json.load(open(summary_file))
    out = {}
    for c in s["metric_per_case"]:
        vals = [m["Dice"] for m in c["metrics"].values() if m.get("n_ref", 1) > 0]
        if vals:
            out[os.path.basename(c["reference_file"])] = sum(vals) / len(vals)
    return out, s["foreground_mean"]["Dice"]


def compare(team, env_pt, env_sc):
    import random
    rows = {}
    for name, d in [(f"{team} (pretrained)", find_result_dir(env_pt, False)),
                    ("ResEnc-L (from scratch)", find_result_dir(env_sc, True))]:
        f = d / f"fold_{FOLD}" / "validation" / "summary.json" if d else None
        if f and f.exists():
            rows[name] = per_case_dice(f)
    log("Results on the held-out test cases")
    for name, (pc, fg) in rows.items():
        print(f"  {name:45s} mean per-case Dice {sum(pc.values()) / len(pc):.3f} on {len(pc)} cases   "
              f"(nnU-Net foreground mean: {fg:.3f})")
    if len(rows) == 2:
        (a, _), (b, _) = rows.values()
        keys = sorted(set(a) & set(b))
        d = [a[k] - b[k] for k in keys]
        rnd = random.Random(0)
        boots = sorted(sum(d[rnd.randrange(len(d))] for _ in d) / len(d) for _ in range(10000))
        mean = sum(d) / len(d)
        print(f"\n  Pretrained - scratch: {mean:+.3f} Dice  (paired bootstrap 95% CI "
              f"{boots[249]:+.3f} to {boots[9749]:+.3f}; pretrained better in "
              f"{sum(x > 0 for x in d)}/{len(d)} cases)")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--team", default=PLAN_TEAM, choices=TEAMS,
                    help="SSL3D ResEnc-L submission to fine-tune (folder name in the Hugging Face repository)")
    ap.add_argument("--nnunet-ref", default="master", help="branch, tag or commit of the public nnU-Net repo")
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--skip-scratch", action="store_true", help="only fine-tune the pretrained model")
    ap.add_argument("--num-proc", type=int, default=8)
    ap.add_argument("--no-compile", action="store_true", help="disable torch.compile")
    args = ap.parse_args()

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if not args.skip_install:
        install_dependencies(args.nnunet_ref)
    env = setup_env(workdir)
    if args.no_compile:
        env["nnUNet_compile"] = "f"
    np_ = str(args.num_proc)

    # 1) data: download MSD Task01 and convert to nnU-Net format (the 4D images are split into four channels)
    if not (Path(env["nnUNet_raw"]) / DATASET_NAME / "dataset.json").exists():
        msd = download_msd(workdir)
        log("Converting to nnU-Net format")
        run(["nnUNetv2_convert_MSD_dataset", "-i", str(msd), "-overwrite_id", str(DATASET_ID), "-np", np_], env)

    # 2) planning: standard base plan (no default preprocessing needed)
    pp = Path(env["nnUNet_preprocessed"]) / DATASET_NAME
    if not (pp / "nnUNetPlans.json").exists():
        log("Fingerprint + planning")
        run(["nnUNetv2_extract_fingerprint", "-d", str(DATASET_ID), "-np", np_], env)
        run(["nnUNetv2_plan_experiment", "-d", str(DATASET_ID)], env)

    # 3) pretrained weights from Hugging Face; plans and preprocessing that match the pretraining
    plan_ckpt = download_checkpoint(workdir, PLAN_TEAM)
    ckpt = plan_ckpt if args.team == PLAN_TEAM else download_checkpoint(workdir, args.team)
    done_marker = pp / f".preprocessed_like_{PRETRAIN_NAME}"
    if find_plans(pp) is None or not done_marker.exists():
        log("Plans and preprocessing like the pretraining (1 mm isotropic, z-score)")
        redo = ["--overwrite_pp"] if find_plans(pp) is not None else []  # an earlier run was interrupted
        run(["nnUNetv2_preprocess_like_nnssl", "-d", str(DATASET_ID), "-n", PRETRAIN_NAME,
             "-pc", str(plan_ckpt), "-am", "like_pretrained", "-np", np_] + redo, env)
        done_marker.touch()
    plans = find_plans(pp)
    assert plans is not None, f"no ptPlans__{PRETRAIN_NAME}____*.json in {pp}"

    # 4) fixed split
    write_split(env)

    # 5) fine-tune the pretrained model, then the from-scratch baseline (same plans and trainer)
    env_pt, env_sc = results_env(env, workdir, args.team), results_env(env, workdir, "from_scratch")
    jobs = [("pretrained", env_pt, ["-overwrite_ckpt_path", str(ckpt)])]
    if not args.skip_scratch:
        jobs.append(("scratch", env_sc, ["--from_scratch"]))
    for name, job_env, extra in jobs:
        d = find_result_dir(job_env, scratch=name == "scratch")
        if d and (d / f"fold_{FOLD}" / "validation" / "summary.json").exists():
            print(f"{name}: already finished ({d})")
            continue
        resume = ["--c"] if d and any((d / f"fold_{FOLD}").glob("checkpoint_*")) else []
        log(f"Training ResEnc-L ({name}{'' if name == 'scratch' else ', ' + args.team}), {TRAINER}")
        run(["nnUNetv2_train_pretrained", str(DATASET_ID), "3d_fullres", FOLD,
             "-p", plans, "-tr", TRAINER] + extra + resume, job_env)

    # 6) evaluation on the held-out test cases (nnU-Net's fold-0 validation = test split)
    compare(args.team, env_pt, env_sc)


if __name__ == "__main__":
    main()
