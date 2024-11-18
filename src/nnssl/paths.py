#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from loguru import logger

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

nnssl_raw = os.environ.get("nnssl_raw")
nnssl_preprocessed = os.environ.get("nnssl_preprocessed")
nnssl_results = os.environ.get("nnssl_results")

if "rocket_preprocessed" in os.environ:
    nnssl_preprocessed = os.environ["rocket_preprocessed"]
    logger.warning(
        "Detected 'rocket_preprocessed' environment variable. Overwriting default nnssl_preprocessed directory."
    )

if nnssl_raw is None:
    print(
        "nnssl_raw is not defined and nnssl_raw can only be used on data for which preprocessed files "
        "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
        "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
        "this up properly."
    )

if nnssl_preprocessed is None:
    print(
        "nnssl_preprocessed is not defined and nnU-Net can not be used for preprocessing "
        "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how "
        "to set this up."
    )

if nnssl_results is None:
    print(
        "nnssl_results is not defined and nnU-Net cannot be used for training or "
        "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information "
        "on how to set this up."
    )
