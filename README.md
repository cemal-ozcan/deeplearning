# Fashion-MNIST CNN Classification — A Three-Way Ablation Study

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO_NAME/blob/main/Fashion_MNIST_CNN_Projesi.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Course Project for **CENG418 — Deep Learning**
> Author: **Cemal Özcan** (220401094) · Instructor: **Çağdaş Eşiyok**

A comparative deep learning study on the Fashion-MNIST dataset. Instead of training a single model, three models are trained and compared in an **ablation study** to isolate the individual contributions of (i) the convolutional architecture and (ii) data augmentation.

## Key Result

The headline finding contradicts a common assumption: data augmentation **did not help** on this dataset under the chosen settings. The Plain CNN (without augmentation) achieved the best test accuracy.

| Model | Test Accuracy | Parameters | Notes |
|-------|--------------:|-----------:|-------|
| Baseline (Dense NN) | 86.58% | 101,770 | No spatial information |
| **Plain CNN (no augmentation)** | **90.97%** ⭐ | 93,322 | Best model |
| Augmented CNN | 89.14% | 93,322 | Augmentation hurt performance |

**Contribution decomposition:**
- Architecture (Baseline → Plain CNN): **+4.39 points**
- Augmentation (Plain CNN → Augmented CNN): **−1.83 points**

The negative augmentation effect is interpreted in the report as a meaningful empirical finding: when the dataset is already large and well-balanced (60,000 training images) and the model is not overfitting, additional augmentation acts more as noise than regularization.

## How to Run

### Option 1 — Google Colab (recommended, no setup)

Click the **"Open In Colab"** badge at the top of this README. Then in Colab:

1. `Runtime → Change runtime type → GPU (T4)`
2. `Runtime → Run all`

Total training time on a T4 GPU: ~7 minutes (Baseline 15s + Plain CNN ~90s + Augmented CNN ~5min).

### Option 2 — Local Jupyter

```bash
git clone https://github.com/USERNAME/REPO_NAME.git
cd REPO_NAME
pip install -r requirements.txt
jupyter notebook Fashion_MNIST_CNN_Projesi.ipynb
```

A CUDA-capable GPU is recommended but not required; on CPU, training takes roughly 30–40 minutes total.

## Repository Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── .gitignore                         # Files excluded from version control
├── Fashion_MNIST_CNN_Projesi.ipynb    # Main Jupyter notebook (all code + analysis)
├── Fashion_MNIST_CNN_Rapor.pdf        # Full project report (PDF)
└── fashion_mnist_cnn.keras            # Saved trained model (best: Plain CNN)
```

## Methodology Summary

1. **Dataset.** Fashion-MNIST: 70,000 grayscale 28×28 images across 10 clothing classes (60K train / 10K test).
2. **Preprocessing.** Pixel normalization to `[0, 1]`; reshaping to `(28, 28, 1)` for `Conv2D` input.
3. **Three models trained on the same train/val/test split:**
   - **Baseline** — Flatten → Dense(128, ReLU) → Dropout(0.3) → Dense(10, Softmax)
   - **Plain CNN** — 3× (Conv2D + MaxPool) → Dense(64) → Dropout(0.5) → Dense(10, Softmax)
   - **Augmented CNN** — Same architecture as Plain CNN, with `RandomRotation`, `RandomZoom`, `RandomTranslation` as preprocessing layers
4. **Optimization.** Adam with default settings; `sparse_categorical_crossentropy` loss.
5. **Evaluation.** Test accuracy, confusion matrix, per-class precision/recall/F1, error pattern analysis.

## Findings (Quick Highlights)

- The convolutional architecture alone provides a **+4.39 point** improvement over the Dense baseline, while using **fewer parameters** (93K vs. 102K) — clear evidence of the parameter efficiency of CNNs on image data.
- The Plain CNN's training and validation curves remained close throughout training (train ≈93%, val ≈91%), indicating that the model was **not overfitting** even without augmentation.
- The misclassifications are not random: the model consistently confuses visually similar classes (Shirt ↔ T-shirt, Coat ↔ Pullover), confirming that the model has learned meaningful visual features.
- The hardest class is **Shirt** (recall ≈ 0.63), which is plausible since shirts overlap visually with both T-shirts and pullovers in 28×28 grayscale.

For the full discussion, mathematical background, training curves, confusion matrix, and per-figure interpretation, see the accompanying PDF report.

## Author

**Cemal Özcan**
Student ID: 220401094
Course: CENG418 — Deep Learning
Instructor: Çağdaş Eşiyok
Date: May 2026

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Fashion-MNIST dataset is provided by Zalando Research (Xiao, Rasul, & Vollgraf, 2017).
- Implemented with TensorFlow / Keras.
