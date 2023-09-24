## Hand Gesture Recognition with OpenCV and MediaPipe
This is a project for recognizing hand gestures using OpenCV and MediaPipe, and then using them to train the CNN model.

# Installation
1. Install the requirements
`pip install -r requirements.txt`
2. Download the MediaPipe models:
`bash models/download_models.sh`

# Usage
1. Run the program
- `run.sh`
this will run the final_project.py file


# Customization
When running the code, it will first prompt you to record new gestures. It is recommended to record new both for accuracy and to grasp the full scope of the project. Enter 'y' to record new. You will then go through the steps to select what poses will be recorded

When prompted, choose the c option to enter custom mode. You will be prompted to specify the number of hand shapes to record, and then to provide a name for each of the hand shapes you wish to record. Once configured, the system will display instructions on the camera display to guide you through recording each hand gesture for a certain amount of time. The system will then train Alexnet to recognize those gestures, and display live classification results.

Alternatively, choose the d option to run the default mode. This mode will recognize five preset gestures: 'c', 'palm', 'fist', 'pointer', and 'pinkie'. The recognized gestures will be displayed in the Python terminal.

# Credits

This project uses the MediaPipe Hand Tracking module for detecting and tracking hands in real-time video streams. The OpenCV library is used for image processing and drawing bounding boxes around the hands. Additionally, the project leverages machine learning through the use of the AlexNet architecture model to train a custom set of hand gestures.

# License
This project is licensed under the MIT License.

In addition, this project uses the MediaPipe Hand Tracking module, which is licensed under the Apache License 2.0, and the OpenCV library, which is licensed under the BSD 3-Clause License. It also uses the PyTorch library, which is licensed under the BSD 3-Clause License.

# Old Code
As explained in the project report, originally we used a dataset of hand poses to build a network and classify. This did not produce accurate results so we swapped to recording the data of the user

The old_version file contains the following scripts that will not be able to be run without the image data (which was not included per the project). You can see the outputs of these files.

1. preprocess_images.py - took in the 20000 images and processed them to be usable for the next step. Look at NN_training.pdf to see the printout of the notebook
   
2. NN.ipynb - was used to create and train an AlexNet model on the filtered image data. Look at test_images to see how the images would be output. In addtion to these images, a torch data fie was made for training that has not been included
   
3. old_classification.py - this is an older version of our final project. It will try to perform classification from the webcam using the trained network from NN.ipynb. It is not accurate on the webcam data, but after you close the webcam data portion by hitting 'q' on the keyboard, it will display the classification results of the test images which is accurate.

Presentation: https://www.canva.com/design/DAFiRKv8pSc/my8RU9Y6rlbBLhkjld4v_Q/edit?utm_content=DAFiRKv8pSc&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
