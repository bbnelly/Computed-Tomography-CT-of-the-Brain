# Brain CT Scan Classifier: Detecting Aneurysm, Cancer, and Tumor

**A deep learning project for classifying brain CT scans into three categories: Aneurysm, Cancer, and Tumor.**

This repository contains a complete Jupyter Notebook implementation of a medical image classification system using TensorFlow/Keras. It includes data loading from DICOM and JPG files, preprocessing, model building (simple CNN + transfer learning with ResNet50), training, evaluation, and real-world testing on external images.


## Project Overview

- **Goal**: Classify axial brain CT scans into **aneurysm**, **cancer**, or **tumor**.
- **Initial Approach**: Custom Convolutional Neural Network (CNN) → achieved ~30% accuracy on external test images (overfitting on limited data).
- **Improved Approach**: Transfer learning with **ResNet50** pre-trained on ImageNet → expected 80-95%+ accuracy.


## Features

- Supports **DICOM (.dcm)** and **JPG** medical images
- Robust preprocessing: resize to 224x224, channel conversion, normalization
- Single image prediction function
- Batch prediction on folders with classification report & confusion matrix
- Real-world testing on randomly downloaded online images
- Easy switch to powerful transfer learning model

## Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV (cv2)
- PyDICOM
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn

Install dependencies:
```bash
pip install tensorflow opencv-python pydicom numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. Place your dataset in folders: `dataset/aneurysm/`, `dataset/cancer/`, `dataset/tumor/`
2. Run the Jupyter Notebook step-by-step
3. Train the model (simple CNN or ResNet50 transfer learning)
4. Test on external images in a similar folder structure
5. Use the prediction function for new single images

## Results

- **Simple CNN**: ~30% accuracy on external real-world test set (26 images)
- **ResNet50 Transfer Learning**: Expected significant improvement (80-95%+)

## Future Improvements

- Add data augmentation
- Fine-tune ResNet50 layers
- Larger/more diverse dataset
- Web interface (Gradio/Streamlit)
- Multi-class including "normal" scans

## Disclaimer

**This is an educational/experimental project only.**  
It is **not** intended for clinical diagnosis. Always consult qualified medical professionals for real medical imaging interpretation.
