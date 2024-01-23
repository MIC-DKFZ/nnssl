
from pathlib import Path
import os



def main():
    s3_bucket_path = "s3://floy-data/clean-data/external/fiona/mr-head-full/"
    # Defining it as input would probably download the data to the valohai instance?
    #   So maybe define as input directly but 
    print("Hello world from valohai!")

    print("Listing files in s3 bucket")
    if Path(s3_bucket_path).exists():
        print("Path exists")
        content = os.listdir(s3_bucket_path)
        n_samples_in_path = len(content)
        print(f"Number of samples in path: {n_samples_in_path}")
        print(content[:10])
    else:
        print("S3-Path not found")

if __name__ == "__main__":
    main()