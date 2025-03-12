from nnssl.configuration import default_num_processes
from nnssl.experiment_planning.plan_and_preprocess_api import extract_fingerprints


def extract_fingerprint_entry():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        type=int,
        nargs="+",
        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
        "planning and preprocessing for these datasets. Can of course also be just one dataset",
    )
    parser.add_argument(
        "-np",
        type=int,
        default=default_num_processes,
        required=False,
        help=f"[OPTIONAL] Number of processes used for fingerprint extraction. " f"Default: {default_num_processes}",
    )
    parser.add_argument(
        "--clean",
        required=False,
        default=False,
        action="store_true",
        help="[OPTIONAL] Set this flag to overwrite existing fingerprints. If this flag is not set and a "
        "fingerprint already exists, the fingerprint extractor will not run.",
    )
    parser.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        help="Set this to print a lot of stuff. Useful for debugging. Will disable progress bar! "
        "Recommended for cluster environments",
    )
    args, unrecognized_args = parser.parse_known_args()
    extract_fingerprints(args.d, args.np, args.clean, args.verbose)


if __name__ == "__main__":
    extract_fingerprint_entry()
