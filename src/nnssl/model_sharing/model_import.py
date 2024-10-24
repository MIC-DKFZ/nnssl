import zipfile

from nnssl.paths import nnssl_results


def install_model_from_zip_file(zip_file: str):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(nnssl_results)