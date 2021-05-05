# Adversarial-Attacks-on-Autonomous-Driving-Vehicles

The repository consits of notebooks to create generator using several method and testing notebook to test the generator on test dataset. This repository also contains a demo of targeted adversarial attack on 

Requirements : Tensorflow 2.3 , Python 3.7
Clone the repository.
Download the object detector model and classifier model from below link.

Save all the model and their configuration in Models folder inside the root directory of this repo.

Download the carla traffic sign dataset from below link.

Save all the dataset in Dataset/ folder inside the root directory of this repo

There are three ways to create the generator.
i) Creating the generator by alt opt optimization algorithm along the two objective. 
ii) Creating the generator by joint training both objective simultaneously.


Train the GAN model and save the generator .
Use the generator created by GAN for Adversarial Attack Testing.
