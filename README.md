
# Plant Disease Detection Model

## Overview

This project implements a **Plant Disease Detection** model using image data. The model uses deep learning techniques to classify plant leaves into different disease categories or as healthy. By analyzing images of plant leaves, the model identifies common diseases such as **bacterial spot**, **early blight**, **late blight**, **leaf mold**, **septoria leaf spot**, and more.

## Features

- **Multi-class classification**: The model classifies images into different disease categories or detects healthy leaves.
- **Image preprocessing**: Input images are resized and normalized for efficient model training.
- **Convolutional Neural Network (CNN)**: Utilizes a deep learning CNN architecture to extract features and classify images.
- **Real-time predictions**: Allows for real-time detection of plant diseases by uploading images via a GUI or command-line interface.

## Dataset

The dataset used is a collection of **plant leaf images** that include various types of plant diseases along with healthy leaves. Each image is labeled with its respective disease category. The dataset can be obtained from sources like **Kaggle** (e.g., PlantVillage dataset) or other plant disease image repositories.

### Dataset Features:
- Image format: `.jpg` or `.png`
- Categories: Healthy, Bacterial Spot, Early Blight, Late Blight, Leaf Mold, and more.
- Image resolution: 256x256 (or resized to this for training).

## Model Architecture

The model uses a **Convolutional Neural Network (CNN)** with the following layers:

- Convolutional Layers: Extracts features from the input image.
- Max Pooling Layers: Reduces the spatial size of the feature maps.
- Fully Connected Layers: Classifies the features into different categories.

You can modify the model architecture in the `model.py` file.

## Usage

### Training the Model

To train the model using the dataset:

```bash
python train.py --epochs 20 --batch_size 32 --learning_rate 0.001
```

- `--epochs`: Number of training epochs.
- `--batch_size`: Number of samples per batch.
- `--learning_rate`: Learning rate for the optimizer.

The trained model will be saved in the `models/` directory.

### Testing the Model

To test the model on the test dataset or individual images:

```bash
python test.py --model_path models/plant_disease_model.h5 --image_path data/test_image.jpg
```

- `--model_path`: Path to the trained model file.
- `--image_path`: Path to the image you want to classify.



Upload an image of a plant leaf, and the model will predict the disease.

## Results

The model achieves an accuracy of approximately **85-90%** on the test dataset. Confusion matrix and precision-recall metrics are provided to evaluate the model’s performance.

## Contribution

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, feel free to create a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Let me know if you need to customize or add more details to the README!
