import os
import json
import nibabel as nib
from multiprocessing import Pool, cpu_count
import argparse

def check_file(path):
    result = {"path": path, "exists": False, "non_empty": False, "readable": False}
    try:
        if os.path.exists(path):
            result["exists"] = True
            if os.path.getsize(path) > 0:
                result["non_empty"] = True
                nib.load(path)  # Try loading as NIfTI
                result["readable"] = True
    except Exception as e:
        result["error"] = str(e)
    return result

def collect_all_paths(data):
    paths = []
    for dataset in data.get("datasets", {}).values():
        for subject in dataset.get("subjects", {}).values():
            for session in subject.get("sessions", {}).values():
                for image in session.get("images", []):
                    paths.append(image["image_path"])
                    # Also check associated masks
                    masks = image.get("associated_masks", {})
                    paths.extend(masks.values())
    return paths

def main(json_path, num_cpus):
    with open(json_path, "r") as f:
        data = json.load(f)

    paths = collect_all_paths(data)
    unique_paths = list(set(paths))  # Remove duplicates

    print(f"Checking {len(unique_paths)} files using {num_cpus} CPU(s)...")

    with Pool(num_cpus) as pool:
        results = pool.map(check_file, unique_paths)

    corrupted_or_missing = [r for r in results if not r["readable"]]
    print(f"\nFound {len(corrupted_or_missing)} corrupted/missing files:")
    for r in corrupted_or_missing:
        print(f"- {r['path']} | exists: {r['exists']} | non-empty: {r['non_empty']} | error: {r.get('error', 'Unreadable')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if dataset files exist, are non-empty, and readable.")
    parser.add_argument("json_path", type=str, help="Path to the dataset JSON file.")
    parser.add_argument("--cpus", type=int, default=cpu_count(), help="Number of CPUs to use (default: all available).")
    args = parser.parse_args()

    main(args.json_path, args.cpus)