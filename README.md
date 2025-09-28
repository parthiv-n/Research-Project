# Dissertation: Prostate Cancer Lesion Detection from MRI with Deep Learning

This project explores deep learning for prostate cancer detection and localisation using multiparametric MRI (mpMRI). Prostate cancer diagnosis currently relies on invasive biopsies and subjective MRI interpretation, both of which are prone to variability and diagnostic pitfalls. Automated methods can improve accuracy, reproducibility, and reduce unnecessary procedures.

Building on advances in U-Net segmentation networks, ordinal Gleason group estimation, Combiner/HyperCombiner networks for multimodal MRI, and evaluation pitfalls of voxel-level metrics, this project:
- Preprocessed T2-weighted, ADC, and DWI scans into isotropic voxel resolution.
- Implemented a 3D U-Net architecture in PyTorch with custom dataloaders and validation pipelines.
- Trained models for lesion segmentation and evaluated performance using Dice similarity and lesion-level detection metrics.
- Investigated how preprocessing, loss functions, and evaluation criteria impact model robustness and clinical relevance.

This work demonstrates the feasibility of using deep learning to segment prostate cancer lesions from mpMRI and highlights the importance of evaluation choices when translating models into clinically meaningful tools.

#order of code:
#1 - transforming_with_glob
#2 - resizing to 1x1x1mm3
#3 - dataloader
#4 - training_with_pytorch
#5 - test_UNET

#patient data is not available here
