# nnssl

WIP library for Self-Supervised Learning of 3D medical image segmentation.
More coming soon


### ToDo's
Current stages of process


- [x] Filter out some images from the dataset

**Error**: cannot load 42k files into docker --> Need to zip.
- [x] Create Zipping of Files
  - [x] Split data into batches that are zipped
  - [ ] Save and upload zipped files
- [ ] Test loading of zipped files for training
  - [ ] Test merging of zipped files
  - [ ] Test `dataset.json` loaded correctly

**Training and integrating with Consti**

- [ ] Test checkpoints and outputs are saved properly and useable for Consti
