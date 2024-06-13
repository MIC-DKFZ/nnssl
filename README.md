# fiona
Building a foundation model for 3D radiological images.

## ToDo's for big valohai trainings
- [x] Save checkpoint every 50 epochs
- [ ] Try torch.compile for some more speed-ups
- [ ] Check if batchnorms can be made faster


### Config for MF big pre-training
- Current ResEnc 160^3 architecture as previously set
- [ ] 3e2/1e2 To Be determined
- [ ] Efficient vs Full (depending on epoch times)
- [ ] Dynamic Masking ratio 60 - 90%
- [ ] N_Epochs: 4k
- [ ] Total 48 -- 16 (N) * 3 or 8 * 6 (N x Batch Size) depending on H100 vs 16 A100s
-