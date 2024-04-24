# nnssl

WIP library for Self-Supervised Learning of 3D medical image segmentation.
More coming soon


### ToDo's
Current stages of process

- [ ] Evaluate, how Lightweight the decoder can become
  - [ ] Test if we can e.g. halve the number of channels in the decoder
  - [ ] Also test how much VRAM is allocated to the decoder overall (if it is low then why bother)
- [ ] Test more efficient Densification
  - [ ] Check epoch-times
  - [ ] Check performance
- [ ] Test higher LR for BS7
  - [x] 3e-2 (ongoing)
  - [x] 5e-2 (ongoing)
  - [ ] Evaluate outcome

**Training and integrating with Consti**

- [ ] Test checkpoints and outputs are saved properly and useable for Consti
