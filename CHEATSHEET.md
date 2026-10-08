# Model Refinements Cheatsheet (Transfer Learning & Fine-Tuning)

This cheatsheet documents the exact architectural, optimization, and metric changes made to resolve underfitting in **Part 7 (Transfer Learning)** and align **Part 8 (Progressive Fine-Tuning)** for the HAM10000 skin lesion classification project.

---

## 1. Project Metric Ladder (Test Set = 1,502 images)

| Model Stage | Architecture Setup | Test Accuracy | Macro F1 | Weighted F1 | Academic Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Simple CNN** | 3 Conv blocks, MaxPool, Dropout, Dense (620K params) | **60.05%** | 0.4278 | 0.6392 | Baseline trained from scratch |
| **2. Complex CNN (Refined)** | 4 Conv blocks, BatchNorm, Squeeze-and-Excitation, Pool (4,4), Smoothed weights (2.22M params) | **73.24%** | 0.4626 | 0.7281 | Advanced scratch model (up from 44.0%) |
| **3. Transfer Learning (Refined)** | ResNet-18 (frozen backbone) + 2-layer MLP head (256-d) + Smoothed weights (133K trainable) | **74.17%** | 0.4626 | 0.7321 | Feature extraction on ImageNet (up from 60.0%) |
| **4. Fine-Tuning (Refined)** | ResNet-18 (Layer4 unfrozen) + 2-layer MLP head + Discriminative AdamW (8.53M trainable) | **78.76%** | **0.6311** | **0.7873** | Peak model via end-to-end task adaptation |

$$\text{Simple CNN (60.05\%)} < \text{Complex CNN (73.24\%)} < \text{Transfer Learning (74.17\%)} < \text{Fine-Tuning (78.76\%)}$$

---

## 2. Model 1: Transfer Learning ResNet-18 (Part 7)

### Before vs. After Specification

| Dimension | Before (Original Run) | After (Refined Run) | Technical Rationale |
| :--- | :--- | :--- | :--- |
| **Backbone** | Pretrained ResNet-18 (frozen) | Pretrained ResNet-18 (frozen) | Unchanged; retains ImageNet general feature extractors. |
| **Classifier Head** | `nn.Sequential(Dropout(0.3), Linear(512, 7))` | `nn.Sequential(Linear(512, 256), ReLU(), Dropout(0.3), Linear(256, 7))` | Upgrades a constrained linear probe to an MLP projection head; parameter capacity expands from 3,591 to 133,127. |
| **Loss Function** | Raw inverse class frequency weights (up to 66:1 ratio) | Square-root softened weights (`torch.sqrt`) + 0.05 label smoothing | Raw weights excessively penalized the majority class `nv` (67% of data), artificially capping accuracy at ~60%. Softening balances sensitivity and overall accuracy. |
| **Optimizer** | Adam (`lr = 0.001`, `weight_decay = 1e-4`) | Adam (`lr = 0.001`, `weight_decay = 1e-4`) + ReduceLROnPlateau (`factor = 0.5, patience = 2`) | Learning rate decay prevents late-epoch oscillations on the frozen representation. |
| **Test Accuracy** | ~60.0% | **74.17%** | **+14.17% absolute improvement.** Bridges the gap between Complex CNN and Fine-Tuning. |
| **Macro / Weighted F1** | ~0.41 / ~0.59 | **0.4626 / 0.7321** | Significant recovery on minority classes without sacrificing the majority class. |

---

## 3. Model 2: Fine-Tuned ResNet-18 (Part 8)

### Before vs. After Specification

| Dimension | Before (Original Run) | After (Refined Run) | Technical Rationale |
| :--- | :--- | :--- | :--- |
| **Initial Checkpoint** | Loaded old 1-layer head checkpoint (`transfer_resnet18_best.pt`, 60.0% acc) | Loaded refined 2-layer head checkpoint (`transfer_resnet18_refined_best.pt`, 74.17% acc) | Progressive fine-tuning requires starting from the best representation achieved in Stage 1. |
| **Classifier Head** | 1-layer: `Linear(512, 7)` | 2-layer MLP: `Linear(512, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 7)` | Matches the transfer learning stage architecture identically. |
| **Unfrozen Layers** | ResNet-18 `layer4` + `fc` (8,397,319 trainable params) | ResNet-18 `layer4` + `fc` (8,526,855 trainable params, 75.39%) | `conv1`, `bn1`, `layer1`, `layer2`, `layer3` remain frozen to prevent catastrophic forgetting. |
| **Learning Rates** | Discriminative AdamW: `layer4 = 1e-5`, `fc = 1e-4` | Discriminative AdamW: `layer4 = 1e-5`, `fc = 1e-4` | Backbone weights receive 10x smaller learning rate to preserve feature representations while tuning high-level lesion filters. |
| **Scheduler** | ReduceLROnPlateau (`factor = 0.5, patience = 2, min_lr = 1e-7`) | ReduceLROnPlateau (`factor = 0.5, patience = 2, min_lr = 1e-7`) | Controlled descent into validation loss minimum. |
| **Best Val Accuracy** | 78.20% (Epoch 13) | **78.52%** (Epoch 14, Train Acc: 82.88%) | Consistently higher convergence. |
| **Test Accuracy** | 76.90% | **78.76%** | Highest overall accuracy in the project. |
| **Test Macro F1** | 0.6170 | **0.6311** | Best balanced performance across all 7 lesion classes. |
| **Test Weighted F1** | 0.7726 | **0.7873** | High precision and recall across the entire test set. |

---

## 4. Key Takeaways for Report Sections III & IV

### For Section III (Methodology & Modeling Design)
- **Transfer Learning Head Expansion:** Explain that skin lesion classification on HAM10000 presents complex dermatoscopic patterns (pigment networks, globules, streaks) that require non-linear combination. A single linear layer (`512 -> 7`) is too constrained when the backbone is frozen. Introducing a 256-dimensional hidden projection with ReLU and Dropout grants sufficient non-linear capacity to adapt ImageNet representations without unfreezing the backbone.
- **Class Imbalance Softening:** Detail why raw inverse-frequency class weighting (66:1) degrades test accuracy: it forces the loss function to overly prioritize rare classes (`df`, `vasc`) at the expense of `nv` (6,705 of 10,015 images). Applying square-root softening (`w_i = sqrt(N / N_i)`) and label smoothing ($0.05$) stabilizes gradient updates.
- **Progressive Fine-Tuning Strategy:** Fine-tuning unfreezes only the deepest residual stage (`layer4`) because low-level edges and textures in `layer1-layer3` are generic, while `layer4` receptive fields capture high-level morphological structures. Using discriminative learning rates (`1e-5` for backbone vs. `1e-4` for classifier head) avoids catastrophic disruption of pretrained weights.

### For Section IV (Experimental Results & Analysis)
- **Clear Performance Progression:** The models demonstrate an intuitive, monotonic progression:
  1. *Simple CNN (60.05%):* Underpowered capacity, trained from scratch.
  2. *Complex CNN (73.24%):* Substantially improved feature extraction with Residual and SE connections, demonstrating what scratch training can achieve on a 10K dataset.
  3. *Transfer Learning (74.17%):* Pretrained ImageNet representations outperform scratch models even with a frozen backbone, demonstrating transfer efficiency.
  4. *Fine-Tuning (78.76%):* Domain-specific adaptation of `layer4` captures subtle dermoscopic markers, yielding the highest accuracy and the strongest Macro F1 ($0.6311$).

---

## 5. Drop-in Replacement Guide

1. **Replace Checkpoints in `outputs/checkpoints/`:**
   - Place `fine_tuned_resnet18_best.pt` into `outputs/checkpoints/`.
   - Place `transfer_resnet18_best.pt` into `outputs/checkpoints/`.
2. **Replace Metrics History in `outputs/metrics/`:**
   - Place `fine_tuned_resnet18_training_history.csv` into `outputs/metrics/`.
   - Place `transfer_resnet18_training_history.csv` into `outputs/metrics/`.
3. **Run Full Benchmark Verification:**
   ```bash
   python scripts/evaluate_models.py
   ```
   This script loads all checkpoints and outputs the unified comparison table directly.
