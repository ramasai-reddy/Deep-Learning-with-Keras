# 🚗 Car Damage Severity Assessment using Deep Learning

## 📌 Project Overview
This project focuses on building a deep learning–based image classification system to assess **car damage severity** from images. The goal is to classify vehicle damage into **three severity categories** using a convolutional neural network with transfer learning.

This repository demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation, and inference, implemented in **Python** using **TensorFlow/Keras**.

---

## 🎯 Objective
The primary objective is to:
- Automatically classify car damage severity into **three classes** based on image data
- Evaluate the effectiveness of **transfer learning** for visual damage assessment
- Provide a reproducible and extensible baseline for further improvements

---

## 🧠 Model Architecture
- **Base model:** VGG16 (pretrained on ImageNet)
- **Approach:** Transfer learning
- **Custom head:**
  - Flatten layer
  - Fully connected (Dense) layers
  - Softmax output layer (3 classes)

The base CNN layers are initially frozen to leverage pretrained visual features, with plans for fine-tuning in later stages.

---

## 🗂️ Dataset
- Image-based dataset organized using directory structure compatible with `ImageDataGenerator`
- Each subdirectory represents a damage severity class
- Images resized to match VGG16 input requirements

> ⚠️ Note: The dataset itself is not included in this repository. Please place your dataset locally and update the paths accordingly.

---

## ⚙️ Training Configuration
- **Loss function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Evaluation metric:** Accuracy
- **Batch size:** Configurable
- **Initial epochs:** 5 (for exploratory analysis)

Data augmentation techniques were applied to improve generalization.

---

## 📊 Results Summary
- Training and validation accuracy showed a **gradual upward trend** during initial epochs
- Validation accuracy peaked at approximately **37–39%**, indicating early learning
- No significant overfitting observed at this stage

### Interpretation
These results suggest that the model has started learning meaningful features but is still **underfitting**, which is expected given:
- Limited number of training epochs
- Complexity of distinguishing visually similar damage categories

---

## 🔍 Limitations
- Limited training epochs prevented full convergence
- Possible class imbalance in the dataset
- Damage severity classification is a visually subtle and challenging task

---

## 🚀 Future Improvements
Potential enhancements include:
- Increasing epochs with **EarlyStopping**
- Fine-tuning deeper layers of the pretrained model
- Stronger data augmentation
- Exploring advanced architectures (ResNet, EfficientNet)
- Addressing class imbalance using class weights

---

## 🛠️ Environment Setup
### Requirements
- Python 3.12
- TensorFlow (CPU)
- NumPy
- Matplotlib
- Pillow

Example setup:
```bash
pip install tensorflow-cpu numpy matplotlib pillow
```

---

## ▶️ How to Run
1. Clone the repository
2. Set up a Python environment (recommended: conda or venv)
3. Place the dataset in the expected directory structure
4. Open the notebook and run all cells

---

## 📁 Repository Structure
```
├── Car_Damage_Severity.ipynb
├── README.md
└── dataset/
    ├── class_1/
    ├── class_2/
    └── class_3/
```

---

## 📌 Conclusion
This project demonstrates the feasibility of using transfer learning for car damage severity classification. While the initial accuracy is moderate, the learning trends indicate strong potential for improvement through extended training and model optimization.

---

## 🙌 Acknowledgements
- TensorFlow / Keras
- ImageNet pretrained models

---

## 📬 Contact
Feel free to open an issue or submit a pull request for improvements or suggestions.

---

⭐ If you find this project useful, consider starring the repository!

