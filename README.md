# Document Type Classification using Deep Learning

## Introduction
In this project, I developed a Document Type Classification system using Deep Learning. 
The model takes a document image as input and predicts the type of document it belongs to. 
This project helped me understand how computer vision techniques can be applied to real-world document processing problems.

---

## Project Objective

The main objective of this project is to build a deep learning model that can automatically classify document images into different categories based on their visual structure and layout.

The model learns patterns from document images and predicts the correct document type without requiring manual intervention.

---

## Dataset

For this project, I used  The RVL-CDIP Dataset, which is one of the most popular datasets for document image classification tasks.

### Dataset Details

- Total Images: Approximately 400,000
- Training Images: 319,999 (Used 5000)
- Validation Images: 40,000 (Used 675)
- Test Images: 40,000 (Used 675)
- Number of Classes: 16

The dataset contains different types of real-world scanned documents and provides a good benchmark for training document classification models.

The dataset was loaded using the Hugging Face datasets library.

```python
from datasets import load_dataset

dataset = load_dataset("chainyo/rvl-cdip")
```

---

## Document Categories

The model classifies documents into the following categories:

- Letter
- Form
- Email
- Handwritten
- Advertisement
- Scientific Report
- Scientific Publication
- Specification
- File Folder
- News Article
- Budget
- Invoice
- Presentation
- Questionnaire
- Resume
- Memo

---

## Approach

Instead of training a neural network from scratch, I used a transfer learning approach.

A pretrained EfficientNet-B3 model was selected and fine-tuned on the RVL-CDIP dataset. Using a pretrained model helps reduce training time and allows the model to learn useful features more efficiently.

The final classification layer of the model was modified so that it could predict the 16 document categories present in the dataset.

---

## Data Preprocessing

Before training, the document images were preprocessed to make them suitable for the model.

The following steps were applied:

- Convert grayscale images to RGB format
- Resize all images to 224 × 224 pixels
- Apply basic data augmentation techniques
- Normalize image values

Data augmentation techniques such as random rotation and horizontal flipping were used to improve the model's ability to generalize.

---

## Model Used

The project uses **EfficientNet-B3**, a popular convolutional neural network architecture known for its strong performance and efficient use of parameters.

### Why EfficientNet-B3?

I selected EfficientNet-B3 because:

- It provides good classification performance.
- It is computationally efficient.
- It works well with transfer learning.
- It offers a good balance between accuracy and training time.

---

## Training Configuration

The model was trained using the following configuration:

| Parameter | Value |
|------------|---------|
| Model | EfficientNet-B3 |
| Image Size | 224 × 224 |
| Batch Size | 64 |
| Epochs | 20 |
| Optimizer | AdamW |
| Learning Rate | 0.0003 |
| Loss Function | Cross Entropy Loss |
| Scheduler | Cosine Annealing |

### Additional Techniques

The following techniques were used during training:

- Transfer Learning
- Mixed Precision Training (AMP)
- Early Stopping
- Gradient Clipping
- Model Checkpointing

---

## Training Environment

The project was trained using Kaggle GPU resources.

### Tools and Libraries

- Python
- PyTorch
- TorchVision
- Hugging Face Datasets
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

### Hardware

- Kaggle Notebook Environment
- NVIDIA Tesla T4 GPU

---

## Current Results

After the initial training epochs, the model achieved the following performance:

| Metric | Value |
|----------|---------|
| Training Accuracy | 90.42% |
| Validation Accuracy | 90.88% |
| Training Loss | 0.8247 |
| Validation Loss | 0.8051 |

These results indicate that the model is learning useful document features and is able to generalize well on unseen validation data.

> Note: Final test metrics will be updated once the complete training and evaluation process is finished.

---

## Challenges Faced

While working on this project, I faced a few practical challenges:

### Large Dataset Size

The RVL-CDIP dataset contains approximately 400,000 images. Downloading, loading, and training on such a large dataset required significant time and computational resources.

### Similar Document Layouts

Some document categories have very similar structures. For example:

- Letter and Memo
- Form and Questionnaire
- Scientific Report and Scientific Publication

This makes classification more challenging and requires the model to learn subtle differences between classes.

### Training Time

Training on the complete dataset takes several hours even when using GPU acceleration.

## Key Learnings

This project helped me gain practical experience in:

- Deep Learning
- Computer Vision
- Transfer Learning
- PyTorch
- Working with large datasets
- Model training and evaluation
- GPU-based model development

It also improved my understanding of how document classification systems are built and used in real-world applications.

---

## Future Improvements

Some improvements that can be explored in the future include:

- Vision Transformer (ViT) models
- Advanced data augmentation techniques
- OCR-assisted document classification
- Hyperparameter tuning
- Ensemble learning approaches

These improvements may help further improve classification performance.

---

## Conclusion

This project demonstrates how deep learning can be used to automatically classify document images into different categories. By using EfficientNet-B3 and transfer learning, a document classification pipeline was developed on the RVL-CDIP dataset.

Working on this project provided valuable hands-on experience in computer vision, model training, transfer learning, and handling large-scale datasets. It also highlighted the importance of proper preprocessing, model selection, and evaluation when building machine learning solutions.

---

