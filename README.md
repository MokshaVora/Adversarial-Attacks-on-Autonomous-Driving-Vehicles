# Adversarial-Attacks-on-Autonomous-Driving-Vehicles

The repository consits of notebooks to create generator using several method and testing notebook to test the generator on test dataset. This repository also contains a demo of targeted adversarial attack on 

Requirements : Tensorflow 2.3 , Python 3.7

### Steps to reproduce the results:
1. Clone the repository.
2. Download the object detector weights and configuration file from: https://pjreddie.com/darknet/yolo/  and put in 'Models' folder
3. Use classifier model 'model.h5' from: https://drive.google.com/drive/folders/14RPRSCZJjYF0VZ-thy-JCeelFlmJnre_?usp=sharing or train using traffic_sign_classifier.py
4. Save all the model and their configuration in 'Models' folder inside the root directory of this repo.
5. Download the carla traffic sign dataset from: https://drive.google.com/drive/folders/14RPRSCZJjYF0VZ-thy-JCeelFlmJnre_?usp=sharing
6. Save all the dataset in Dataset/ folder inside the root directory of this repo
7. There are three ways to create the generator.
    i) Creating the generator by alt opt optimization algorithm along the two objective. 
    ii) Creating the generator by joint training both objective simultaneously.
8. Train the GAN model and save the generator.
9. Download the images of Carla simulator world from: https://drive.google.com/drive/folders/14RPRSCZJjYF0VZ-thy-JCeelFlmJnre_?usp=sharing
10. Use the generator created by GAN for Adversarial Attack Testing. 
