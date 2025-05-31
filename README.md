🐶🐱 Cats vs Dogs Image Classification

This project demonstrates a progression from a basic CNN to a high-performance image classifier using Transfer Learning with MobileNetV2 to distinguish between cat and dog images.

Objective

To build a binary classifier that distinguishes between images of cats and dogs. Starting with a custom CNN and moving to a pre-trained transfer learning model, this project showcases the difference in performance between the two approaches.

Dataset

Source: Manually curated dataset.

Structure:

S37 - dataset.zip
│
└───dataset
    ├───training_set
    │   ├───cats
    │   └───dogs
    └───test_set
        ├───cats
        └───dogs

8000 training images and 2000 test images

Approach

1. Custom CNN Architecture (Baseline)

Built a model with 4 convolutional layers and 4 dense layers.

Resulted in low accuracy (~51%), indicating underfitting and poor generalization.

2. Transfer Learning with MobileNetV2

🔧 Preprocessing:

Used image_dataset_from_directory with normalization (Rescaling) and prefetching via AUTOTUNE for performance.

Image size: 160x160


Model Structure:

MobileNetV2 (pre-trained on ImageNet)

Top layers frozen (trainable=False) to preserve learned features.

Custom classifier added:

Flatten

Dense(128, relu) + Dropout

Dense(1, sigmoid)





Training:

Optimizer: Adam with learning_rate=1e-4

Loss: binary_crossentropy

Metrics: accuracy, AUC

Callback: EarlyStopping (patience=5)




Performance:

Significant improvement in validation accuracy and AUC.

Early stopping ensured optimal weights were retained.




Evaluation

Predictions made using threshold on sigmoid output.

Evaluation included:

Accuracy Score

Confusion Matrix

AUC




Key Learnings

Transfer learning dramatically boosts performance when data is limited.

Lower learning rates help preserve valuable pre-trained features.

AUTOTUNE improves pipeline efficiency.


 
 
 
Next Steps

Fine-tune MobileNetV2 by unfreezing top layers.

Experiment with other pre-trained models like ResNet50, InceptionV3.

Add image augmentation for better generalization.

Convert the model to a web API or interactive demo.





Skills Demonstrated

TensorFlow/Keras modeling

Data preprocessing and pipeline optimization

CNN architecture design

Transfer learning with MobileNetV2

Model evaluation and visualization



Author

Steve Philip - stephenphilip28@gmail.com

