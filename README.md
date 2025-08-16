# Neural Photo Restyler — MVP

A compact, reproducible PyTorch-based inference-first MVP for single-pass image style transfer.
Accepts a content image and either a preset style or a user-uploaded style image, then produces a stylized output using a TorchScript model or a state-dict-loaded Johnson-style generator.

This version targets the pinned environment:
- `python==3.9.9`
- `torch==1.13.1+cu116`
- `torchvision==0.14.1+cu116`
