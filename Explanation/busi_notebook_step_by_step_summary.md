# BUSI FDL Final Notebook - Step-by-Step Simple Summary

This file explains what each notebook step does in simple Bangla-English. It follows the notebook order from Step 0 to Step 19.

## Overall Project Logic

The notebook builds an end-to-end deep learning workflow for BUSI breast ultrasound images.

- First, it prepares and analyzes the dataset.
- Then it splits data and applies augmentation.
- Then it trains classification models to classify images as benign, malignant, or normal.
- Then it trains segmentation models to segment tumor regions for abnormal images.
- Finally, it evaluates all models and builds a full pipeline where classification decides whether segmentation is needed.

Important idea:

> Segmentation is meaningful only when the image is abnormal. So classification is used first, then segmentation is applied to benign/malignant cases.

## Step 0: Google Colab Setup

### 0A: Google Drive Mount

Google Drive is mounted so the notebook can access the BUSI dataset, save augmented files, save CSV splits, and save trained model checkpoints.

### 0B: GPU Check

The notebook checks whether GPU/CUDA is available. In the saved run, device was `cuda`.

### 0C: Dataset Path Verify

The dataset folder path is checked to make sure the required dataset directory exists before training starts.

### 0D: Install Extra Libraries

Extra libraries needed for the project are installed, such as libraries for pretrained models, augmentation, metrics, and visualization.

## Step 1: Libraries Import

All required Python libraries are imported.

Main categories:

- Deep learning: PyTorch, TorchVision, timm
- Data handling: pandas, numpy, os, json
- Image processing: cv2, PIL, albumentations
- Metrics: sklearn metrics, scipy distance tools
- Visualization: matplotlib, seaborn
- Utility: random seeds, tqdm, collections

## Step 2: Configuration & Hyperparameters

This step defines all important project settings.

Main configuration:

- Base path: `/content/drive/MyDrive/BUSI_Project`
- Classes: `benign`, `malignant`, `normal`
- Number of classes: `3`
- Image size: `256 x 256`
- Split ratio: `70 / 15 / 15`
- Classification batch size: `16`
- Segmentation batch size: `16`
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR`
- Early stopping patience: `10`
- Seed: `42`

Classification training:

- Phase 1: 10 epochs, LR `1e-3`
- Phase 2: 10 epochs, LR `1e-4`
- Phase 3: 30 epochs, LR `1e-5`
- Total planned epochs: 50

Segmentation training:

- Phase 1: 5 epochs, LR `1e-3`
- Phase 2: 5 epochs, LR `1e-4`
- Phase 3: 40 epochs, LR `1e-5`
- Total planned epochs: 50

Ensemble thresholds:

- Classification normal threshold: `0.4`
- Segmentation mask threshold: `0.5`

Segmentation loss:

- BCE + Dice combo loss
- Alpha: `0.5`

## Step 3: Dataset Analysis (EDA) + OR Operation

### 3A: EDA - Class Distribution

The notebook counts original images in each class.

Result:

| Class | Images |
|---|---:|
| benign | 437 |
| malignant | 210 |
| normal | 133 |
| total | 780 |

Purpose:

- Understand class imbalance.
- Show dataset overview.
- Prepare for class weighting and augmentation.

### 3B: EDA - Image Size Distribution

The notebook checks image dimensions across the dataset.

Purpose:

- See whether all images have same size or different sizes.
- Confirm resizing is needed before model input.

### 3C: EDA - Mask Quality Check

The notebook checks mask files and mask validity.

Purpose:

- Make sure tumor masks exist where required.
- Identify mask-related issues before segmentation training.

### 3D: EDA - Corrupt Image Check

The notebook tries to read images to detect corrupt/unreadable files.

Purpose:

- Prevent training crashes due to broken image files.

### 3E: EDA - Sample Images & Masks Visualization

The notebook visualizes sample ultrasound images and masks.

Purpose:

- Qualitative dataset inspection.
- Show image-mask relationship for presentation.

### 3F: EDA - Mask Coverage Analysis

The notebook calculates how much area the mask covers in images.

Purpose:

- Understand lesion size distribution.
- Check whether masks are too small, too large, or abnormal.

### 3G: Merged Data Preparation

If one image has multiple mask files, the notebook merges masks using OR operation.

Simple meaning:

> If any mask marks a pixel as tumor, the merged mask also marks that pixel as tumor.

Purpose:

- Create one final mask per image.
- Make segmentation training easier and consistent.

### 3H: Duplicate Check (Merged Dataset)

The notebook checks duplicate images after merged data preparation.

Purpose:

- Identify repeated images or possible dataset issues.

## Step 4: Split the Dataset

### 4A: Image Path & Label Collect

The notebook collects all image paths and their class labels.

Purpose:

- Prepare input list for train/validation/test split.

### 4B: Stratified Split (70/15/15)

The dataset is split using stratified splitting.

Result:

- Train: 546 images
- Validation: 117 images
- Test: 117 images

Stratified means:

> Class ratio is preserved as much as possible in train, validation, and test sets.

### 4C: Split Result Print

The notebook prints class-wise split distribution.

Train:

- benign: 306
- malignant: 147
- normal: 93

Validation:

- benign: 65
- malignant: 32
- normal: 20

Test:

- benign: 66
- malignant: 31
- normal: 20

### 4D: Splitted Data Save in CSV

Train, validation, and test split information is saved as CSV files.

Purpose:

- Keep split fixed and reproducible.
- Avoid accidental different split each run.

### 4E: Split Visualization

The notebook plots split distribution.

Purpose:

- Show data split clearly in report/presentation.

## Step 5: Offline Augmentation

### 5A: Offline Augmentation Transform

The notebook defines offline augmentation transforms.

Used for training data only:

- Horizontal flip
- Vertical flip
- Rotation around +/- 15 degrees
- Elastic transform with alpha `50`, sigma `5`

Purpose:

- Increase training data.
- Reduce class imbalance.
- Improve model generalization.

### 5B: Train CSV Load + Augment Count

The notebook checks train class count before augmentation:

- benign: 306
- malignant: 147
- normal: 93

Augmentation plan:

- benign: `0` extra copy per image
- malignant: `1` augmented copy per image
- normal: `2` augmented copies per image

Reason:

> Benign already has the most images, so malignant and normal are augmented more.

### 5C: Augmentation Function

The notebook defines a function to apply augmentation to image and mask together.

Important:

- Image and mask must get the same geometric transformation.
- Otherwise mask would not match the image.

### 5D: Run Augmentation

The notebook actually creates augmented training files.

Purpose:

- Generate new augmented files only for training set.
- Validation and test are not augmented offline.

### 5E: Train CSV Update

The train CSV is updated to include augmented file paths and labels.

Purpose:

- DataLoader can read both original and augmented train samples.

### 5F: Final Train Count

After augmentation, train set becomes:

- benign: 306
- malignant: 294
- normal: 279
- total: 879

### 5G: Augmentation Visualization

The notebook visualizes before/after augmentation examples.

Purpose:

- Verify augmentation looks reasonable.
- Use as qualitative explanation in presentation.

## Step 6: Class Weight Calculate

### 6A: Updated Train CSV Load

The updated train CSV is loaded after augmentation.

Counts:

- benign: 306
- malignant: 294
- normal: 279
- total: 879

### 6B: Class Weight Calculate

The notebook calculates class weights using:

```text
weight = total_samples / (NUM_CLASSES * class_count)
```

Results:

- benign: `0.9575`
- malignant: `0.9966`
- normal: `1.0502`

Simple meaning:

> Smaller classes get slightly higher loss importance.

### 6C: Class Weights -> PyTorch Tensor

Class weights are converted into a PyTorch tensor.

Purpose:

- Use weights inside `CrossEntropyLoss`.

### 6D: Balance Check

The notebook checks imbalance ratio.

Result:

- max class: 306
- min class: 279
- imbalance ratio: 1.10
- status: well balanced

Class weights are still used as safety measure.

### 6E: Save Class Weights

Class weights are saved to `class_weights.json`.

Purpose:

- Reuse exact weights later.

### 6F: Class Weight Visualization

The notebook plots train class distribution and class weights.

## Step 7: Dataset Class & DataLoader

### 7A: On-the-fly Augmentation Transforms

The notebook defines transforms applied during each epoch.

Training transforms include:

- GaussNoise with variance range 10-50
- RandomBrightnessContrast +/- 0.2
- GaussianBlur kernel 3-5
- CLAHE clip limit 2.0
- Resize to 256 x 256
- ImageNet normalization

Validation/test transforms:

- Resize to 256 x 256
- ImageNet normalization
- No geometric augmentation

### 7B: Classification Dataset Class

The notebook defines a PyTorch dataset for classification.

It returns:

- image tensor
- class label

### 7C: Segmentation Dataset Class

The notebook defines a PyTorch dataset for segmentation.

It returns:

- image tensor
- mask tensor

Important:

> Segmentation dataset uses abnormal images with masks, not normal images.

### 7D: Create Dataloader

The notebook creates train/validation/test DataLoaders.

Purpose:

- Load data in batches.
- Shuffle training data.
- Feed data to models efficiently.

### 7E: Dataloader Summary

The notebook prints number of images and batches for classification and segmentation DataLoaders.

### 7F: Batch Shape Test

The notebook takes one batch and checks tensor shapes.

Purpose:

- Confirm image and mask dimensions are correct before training.

## Step 8: Classification Models Setup

### 8A: EfficientNet-B4 Model Class

The notebook loads EfficientNet-B4 from TorchVision with ImageNet-1K pretrained weights.

Modified head:

- Dropout `0.4`
- Linear layer to `3` output classes

Type:

- CNN-based classification model

### 8B: Swin Transformer Tiny Model Class

The notebook loads Swin-T from TorchVision with ImageNet-1K pretrained weights.

Modified head:

- Dropout `0.3`
- Linear layer to `3` output classes

Type:

- Transformer-based vision model

### 8C: MobileViTv2-100 Model Class

The notebook loads `mobilevitv2_100` from timm with pretrained weights.

Modified head:

- Dropout `0.3`
- Linear layer to `3` output classes

Notebook display name:

- MobileViT-S

Actual code model:

- `mobilevitv2_100`

Type:

- Hybrid CNN-Transformer model

### 8D: Build All 3 Classification Models

The notebook builds the three classification models.

Parameter counts:

| Model | Parameters |
|---|---:|
| EfficientNet-B4 | 17,553,995 |
| Swin-T | 27,521,661 |
| MobileViT-S / MobileViTv2-100 | 4,390,380 |

### 8E: Classification Models Summary Table

The notebook prints model type and parameter count.

### 8F: Forward Pass Test

The notebook sends a dummy batch through the models.

Purpose:

- Confirm output shape is correct.
- Expected output: batch size x 3 class logits.

## Step 9: Classification Training Function

### 9A: Freeze/Unfreeze Helper Functions

The notebook defines functions to control trainable layers.

Used for:

- freeze backbone
- partial unfreeze
- full fine-tune

### 9B: Train One Epoch + Validate Functions

The notebook defines functions for:

- one training epoch
- validation pass
- loss calculation
- accuracy calculation

### 9C: 3-Phase Classification Training Function

The notebook defines the full classification training function.

Training strategy:

1. Freeze backbone
2. Partial unfreeze
3. Full fine-tune

Best model is saved based on validation accuracy.

### 9D: Training History Plot Function

The notebook defines function to plot:

- train loss
- validation loss
- validation accuracy

### 9E: Training Setup Summary

Classification setup:

- Freeze 10 epochs
- Partial 10 epochs
- Full 30 epochs
- Total 50 planned epochs
- LR: `0.001 -> 0.0001 -> 1e-05`
- Optimizer: AdamW
- Weight decay: `0.01`
- Loss: weighted CrossEntropyLoss
- Scheduler: CosineAnnealingLR

## Step 10: Train 3 Classification Models

### 10A: Train EfficientNet-B4

EfficientNet-B4 is trained using the 3-phase fine-tuning function.

Best validation accuracy:

- `0.8632`

Best checkpoint:

- `EfficientNet-B4_best.pth`

### 10B: Train Swin-T

Swin-T is trained using the same 3-phase fine-tuning strategy.

Best validation accuracy:

- `0.8803`

Best checkpoint:

- `Swin-T_best.pth`

### 10C: Train MobileViT-S / MobileViTv2-100

MobileViTv2-100 is trained using the same 3-phase fine-tuning strategy.

Best validation accuracy:

- `0.9060`

Best checkpoint:

- `MobileViT-S_best.pth`

### 10D: Load Models Training History from Disk

Training history JSON files are loaded from disk.

Purpose:

- Reuse saved loss/accuracy history without retraining.

### 10E: Summary Table

The notebook prints classification training summary.

Best validation accuracies:

- EfficientNet-B4: 0.8632
- Swin-T: 0.8803
- MobileViT-S: 0.9060

### 10F: Compare Training Curves

The notebook plots training curves for all classification models.

### 10G: Best Model Load from Checkpoints

The notebook loads best saved classification checkpoints:

- `EfficientNet-B4_best.pth`
- `Swin-T_best.pth`
- `MobileViT-S_best.pth`

## Step 11: Classification Ensemble

### 11A: Ensemble Prediction Function Define

The notebook defines weighted soft voting.

Simple meaning:

> Each model outputs class probabilities. The probabilities are averaged using validation accuracy-based weights.

### 11B: Individual Model Predictions (Val Set)

Validation accuracies used as ensemble weights:

- EfficientNet-B4: 0.8632
- Swin-T: 0.8803
- MobileViT-S: 0.9060

### 11C: Ensemble Prediction (Val Set)

The weighted ensemble is evaluated on validation set.

Validation accuracy:

- Ensemble: 0.9060

### 11D: Comparison - Individual vs Ensemble

The notebook compares individual validation results with ensemble validation result.

### 11E: Visualization

The notebook creates a bar chart for validation comparison.

## Step 12: Classification Evaluation

### 12A: Test Set Predictions (All Models)

The notebook predicts test set labels using all individual classification models.

### 12B: Ensemble Predictions (Test Set)

The notebook predicts test set labels using the weighted soft voting ensemble.

### 12C: 7 Metrics Calculate Functions

The notebook defines classification metrics.

Metrics include:

- Accuracy
- Macro AUC-ROC
- Macro F1
- MCC
- Sensitivity per class
- Specificity per class
- Confusion matrix / classification report support

### 12D: All Models + Ensemble Metrics Calculate

The notebook calculates all classification metrics.

### 12E: Metrics Comparison Table

Test set result:

| Metric | EfficientNet-B4 | Swin-T | MobileViT-S | Ensemble |
|---|---:|---:|---:|---:|
| Accuracy | 0.8034 | 0.8547 | 0.8205 | 0.8632 |
| AUC-ROC Macro | 0.9391 | 0.9438 | 0.9260 | 0.9425 |
| Macro F1 | 0.8097 | 0.8481 | 0.8102 | 0.8612 |
| MCC | 0.6829 | 0.7503 | 0.7020 | 0.7729 |

Best classification model:

- Ensemble

### 12F: Confusion Matrix - All Models

The notebook plots confusion matrices.

Purpose:

- Show class-wise errors.
- See which classes are confused.

### 12G: ROC Curves - All Models

The notebook plots ROC curves for models.

Purpose:

- Show probability-based class separation quality.

### 12H: Classification Report + Best Model Highlight

Ensemble classification report:

- accuracy: 0.8632
- macro F1: 0.8612
- MCC: 0.7729
- macro AUC: 0.9425

Best model by accuracy:

- Ensemble

### 12I: Bootstrap Confidence Intervals (Classification)

The notebook calculates confidence intervals for classification metrics using bootstrap resampling.

Ensemble 95% CI examples:

- Accuracy: 0.8638 (0.7949 - 0.9231)
- AUC-ROC Macro: 0.9424 (0.9014 - 0.9766)
- Macro F1: 0.8590 (0.7888 - 0.9212)
- MCC: 0.7726 (0.6726 - 0.8710)

## Step 13: Segmentation Models Setup

### 13A: Attention Gate Block + Conv Block

The notebook defines:

- AttentionGate
- ConvBlock

AttentionGate:

> Filters skip connection features so the decoder focuses more on important lesion regions.

ConvBlock:

> Two Conv3x3 + BatchNorm + ReLU layers.

### 13B: Attention U-Net Model Class

The notebook defines Attention U-Net using ResNet34 encoder.

Two versions are possible:

- pretrained encoder
- from-scratch encoder

Output:

- 1-channel segmentation mask logits

### 13C: U-Net++ (Nested U-Net) Model Class

The notebook defines U-Net++ with ResNet34 encoder and nested decoder nodes.

Simple meaning:

> Instead of direct skip connection only, U-Net++ uses multiple intermediate fusion nodes to reduce semantic gap between encoder and decoder.

### 13D: TransUNet Model Class

The notebook defines TransUNet with:

- ResNet34 CNN encoder
- transformer bottleneck
- U-Net style decoder

Transformer settings:

- 64 tokens from 8 x 8 feature map
- embedding dimension 768
- 6 transformer encoder layers
- 12 attention heads
- FFN dimension 3072
- dropout 0.1

### 13E: Build All 3 Segmentation Models

The notebook builds pretrained segmentation models.

Parameter counts:

| Model | Parameters |
|---|---:|
| Attention-UNet | 24,524,941 |
| UNet++ | 26,155,585 |
| TransUNet | 67,798,657 |

### 13F: Model Summary Table

The notebook prints segmentation model type and parameter count.

### 13G: Forward Pass Test

The notebook sends dummy input through segmentation models.

Purpose:

- Confirm output shape is correct.
- Expected output: batch size x 1 x 256 x 256.

## Step 14: Segmentation Training Function

### 14A: Dice Loss + BCE Dice Combo Loss

The notebook defines segmentation loss.

Loss:

- BCE loss
- Dice loss
- combined using alpha `0.5`

Simple meaning:

> BCE checks pixel-wise correctness, Dice checks mask overlap.

### 14B: Seg Train One Epoch + Validation Functions

The notebook defines segmentation training and validation loops.

Validation metric:

- Dice score

### 14C: Full Segmentation Training Function

The notebook defines 3-phase segmentation training.

Training strategy:

1. Freeze encoder
2. Partial unfreeze
3. Full fine-tune

Best model is saved based on validation Dice.

### 14D: Training History Plot Function + Summary

The notebook defines plots for:

- train loss
- validation loss
- validation Dice

Segmentation setup:

- Freeze encoder 5 epochs
- Partial 5 epochs
- Full 40 epochs
- Total 50 planned epochs
- LR: `0.001 -> 0.0001 -> 1e-05`
- Optimizer: AdamW
- Loss: BCE + Dice combo
- Scheduler: CosineAnnealingLR

## Step 15: Train 3 Segmentation Models

### 15 Scratch: From-Scratch Attention-UNet Training

The notebook trains Attention U-Net from scratch as a comparison experiment.

Important:

- ResNet34 encoder uses random initialization.
- No ImageNet pretrained weights.

Best validation Dice:

- 0.7319

Checkpoint:

- `Attention-UNet_scratch_best.pth`

### 15A: Train Attention-UNet

The notebook trains pretrained Attention U-Net.

Best validation Dice:

- 0.7618

Checkpoint:

- `Attention-UNet_best.pth`

### 15B: Train UNet++

The notebook trains pretrained U-Net++.

Best validation Dice:

- 0.7747

Checkpoint:

- `UNetplusplus_best.pth`

### 15C: Train TransUNet

The notebook trains pretrained TransUNet.

Best validation Dice:

- 0.7705

Checkpoint:

- `TransUNet_best.pth`

### 15D: Load Model Histories from Disk

Segmentation history JSON files are loaded.

Purpose:

- Reuse saved training curves without retraining.

### 15E: Summary Table

The notebook prints best validation Dice for segmentation models.

### 15F: Compare Training Curves

The notebook plots segmentation training curves.

### 15G: Load Best Models from Checkpoints

The notebook loads best pretrained segmentation checkpoints:

- `Attention-UNet_best.pth`
- `UNetplusplus_best.pth`
- `TransUNet_best.pth`

### 15H: Post-Processing Mask Function

The notebook defines mask post-processing.

Post-processing:

- keep largest connected component
- fill holes

Purpose:

- Remove small noisy predicted regions.
- Make final mask cleaner.

## Step 16: Segmentation Ensemble

### 16A: Segmentation Ensemble Prediction Function

The notebook defines segmentation ensemble.

Method:

- each model outputs probability mask
- masks are averaged equally
- threshold 0.5 is applied
- post-processing is applied

Simple formula:

```text
final probability mask = (mask1 + mask2 + mask3) / 3
```

### 16B: Individual Model Predictions (Val Set)

The notebook calculates validation Dice for individual segmentation models.

Also includes from-scratch Attention U-Net validation Dice from history.

### 16C: Ensemble Prediction (Val Set)

The notebook calculates validation Dice for segmentation ensemble.

### 16D: Comparison - Individual vs Ensemble

The notebook compares individual segmentation models and ensemble on validation set.

### 16E: Sample Predictions Visualization

The notebook visualizes predicted masks from segmentation models.

Purpose:

- Qualitative evaluation.
- Show how model predictions look against ground truth mask.

### 16F: Bar Chart + Cleanup

The notebook plots segmentation validation comparison and clears memory.

## Step 17: Segmentation Evaluation

### 17A: Hausdorff Distance 95% Function

The notebook defines HD95 metric.

Simple meaning:

> HD95 measures boundary distance between predicted mask and ground-truth mask. Lower is better.

### 17B: Calculate 6 Segmentation Metrics Function (Per-Image)

The notebook defines segmentation metrics:

- Dice
- IoU
- Pixel Accuracy
- Sensitivity
- Specificity
- Hausdorff95

### 17C: Per-Image Hausdorff + Metrics Function

The notebook combines per-image metrics with HD95 calculation.

### 17D: Evaluate All Individual Models (Test Set)

The notebook evaluates pretrained segmentation models on test set.

### 17E: Ensemble Evaluation (Test Set)

The notebook evaluates segmentation ensemble on test set.

### 17F: Metrics Comparison Table

Test set result:

| Metric | Attention-UNet | UNet++ | TransUNet | Ensemble | A-UNet Scratch |
|---|---:|---:|---:|---:|---:|
| Dice | 0.7602 | 0.7933 | 0.7788 | 0.7770 | 0.7459 |
| IoU | 0.6696 | 0.7035 | 0.6893 | 0.6911 | 0.6571 |
| Pixel Accuracy | 0.9595 | 0.9644 | 0.9627 | 0.9631 | 0.9537 |
| Sensitivity | 0.7859 | 0.8128 | 0.7718 | 0.7837 | 0.7726 |
| Specificity | 0.9816 | 0.9838 | 0.9871 | 0.9852 | 0.9799 |
| Hausdorff95 | 28.6401 | 24.9965 | 26.0162 | 26.8179 | 26.6026 |

Best segmentation model:

- UNet++

### 17G: Bar Chart - Dice Comparison

The notebook plots Dice score comparison.

### 17H: Multi-Metric Bar Chart

The notebook plots multiple segmentation metrics.

### 17I: Best Model Highlight + Cleanup

Best segmentation model:

- UNet++
- Dice: 0.7933
- IoU: 0.7035

### 17J: Bootstrap Confidence Intervals (Segmentation)

The notebook calculates bootstrap confidence intervals for segmentation metrics.

Example Dice 95% CI:

- Attention-UNet: 0.7602 (0.7026 - 0.8110)
- UNet++: 0.7933 (0.7415 - 0.8394)
- TransUNet: 0.7788 (0.7242 - 0.8272)
- Ensemble: 0.7770 (0.7193 - 0.8268)
- Attention-UNet Scratch: 0.7459 (0.6869 - 0.7975)

### 17K: From-Scratch vs Pretrained Comparison

The notebook compares Attention U-Net from scratch with pretrained Attention U-Net.

Result:

| Metric | From Scratch | Pretrained | Difference |
|---|---:|---:|---:|
| Dice | 0.7459 | 0.7602 | +0.0143 |
| IoU | 0.6571 | 0.6696 | +0.0125 |
| Pixel Accuracy | 0.9537 | 0.9595 | +0.0058 |
| Sensitivity | 0.7726 | 0.7859 | +0.0134 |
| Specificity | 0.9799 | 0.9816 | +0.0018 |
| Hausdorff95 | 26.6026 | 28.6401 | +2.0376 |

Validation Dice:

- From scratch: 0.7319
- Pretrained: 0.7618
- Difference: +0.0299

Notebook conclusion:

> Transfer learning helps because BUSI is a small medical imaging dataset.

Important nuance:

> For overlap metrics, pretrained Attention U-Net is better. For HD95 in this table, from-scratch has lower HD95 than pretrained Attention U-Net, but overall Dice/IoU are better for pretrained.

## Step 18: Full Pipeline Integration

### 18A: Load All 6 Models from Checkpoints

This optional block reloads:

Classification:

- EfficientNet-B4
- Swin-T
- MobileViT-S

Segmentation:

- Attention-UNet
- UNet++
- TransUNet

Note:

> This block says "All 6 models" because it loads 3 classification + 3 pretrained segmentation models. The from-scratch Attention U-Net is saved separately and used in comparison/evaluation.

### 18B: Single Image Prediction Function

The notebook defines the full pipeline prediction function.

Process:

1. Input image is loaded.
2. Classification ensemble predicts benign/malignant/normal.
3. If prediction is normal, segmentation is skipped.
4. If prediction is benign or malignant, segmentation model predicts mask.
5. Final output contains class prediction and mask if needed.

### 18C: Visualization Function (with GT Mask)

The notebook defines visualization for:

- input image
- classification result
- predicted mask
- ground-truth mask

### 18D: Full Pipeline Demo (Test Set Samples)

The notebook runs the full pipeline on test samples.

Purpose:

- Demonstrate end-to-end workflow.
- Show classification first, then conditional segmentation.

## Step 19: Final Visualization + Summary Report

### 19A: Classification Metrics Reload (if needed)

The notebook reloads or recalculates classification metrics if variables are missing.

Purpose:

- Make final summary cells runnable even if earlier cells were skipped.

### 19B: Segmentation Metrics Reload (if needed)

The notebook reloads or recalculates segmentation metrics if variables are missing.

### 19C: Dataset Summary

The notebook prints final dataset summary.

Expected summary:

- Original images: 780
- Train split: 546 original -> 879 after augmentation
- Validation: 117
- Test: 117

### 19D: Classification Results Table

The notebook prints final classification table.

Best classification result:

- Ensemble
- Accuracy: 0.8632
- AUC-ROC Macro: 0.9425
- Macro F1: 0.8612
- MCC: 0.7729

### 19E: Segmentation Results Table

The notebook prints final segmentation table.

Best segmentation result:

- UNet++
- Dice: 0.7933
- IoU: 0.7035

### 19F: Model Architecture Summary

The notebook summarizes model types.

Classification:

- EfficientNet-B4: CNN
- Swin-T: Transformer
- MobileViT-S / MobileViTv2-100: Hybrid CNN-Transformer

Segmentation:

- Attention-UNet: U-Net + Attention Gate
- UNet++: Nested U-Net
- TransUNet: CNN + Transformer
- Attention-UNet from scratch: same Attention U-Net architecture, no pretrained weights

### 19G: Training Configuration

The notebook prints final training setup.

Classification:

- Freeze 10 -> Partial 10 -> Full 30
- Loss: weighted CrossEntropyLoss
- Optimizer: AdamW
- Ensemble: weighted soft voting

Segmentation pretrained:

- Freeze encoder 5 -> Partial 5 -> Full 40
- Loss: BCE + Dice combo
- Encoder: pretrained ResNet34 ImageNet-1K
- Ensemble: mask averaging, threshold 0.5

Segmentation from scratch:

- Single phase full training
- ResNet34 random initialization
- No pretrained weights

### 19H: Final Comparison Charts

The notebook creates final visual comparison charts.

Purpose:

- Summarize classification and segmentation performance visually.
- Support presentation discussion.

## Final High-Level Interpretation

Classification:

- Best individual validation model: MobileViT-S / MobileViTv2-100.
- Best final test result: classification ensemble.
- Ensemble accuracy: 0.8632.

Segmentation:

- Best final test segmentation model: UNet++.
- Best Dice: 0.7933.
- Best IoU: 0.7035.
- Segmentation ensemble did not beat UNet++ on Dice.

From-scratch experiment:

- Attention U-Net from scratch was useful as an alternative experiment.
- Pretrained Attention U-Net performed better on Dice and IoU.
- This supports the choice of transfer learning for a small medical dataset.

Main project strength:

> The notebook is a complete end-to-end deep learning workflow: data analysis, splitting, augmentation, classification, segmentation, ensemble, evaluation, confidence intervals, and final integrated pipeline.

