# Neural-Networks--All-of-Whose-Derivatives-are-Self-Normalizing-Neural-Networks

This is a project involving implicit representation learning using neural networks designed so that all their derivatives are bidirectional self-normalizing neural networks, which were introduced by Lu et al. in <I>Bidirectional Self-Normalizing Neural Networks</I>. Another improvement which contributes to the result is a modification of the positional encoder introduced by Mildenhall et al. in their paper <I>NerRF: Representing Scenes as Neural Radiance Fields for View Synthesis</I>.

This method substantially outperforms methods like SIREN on the task of fitting images, being able to achieve PSNR's above 95, i.e. MSE below 10<SUP>-9.5</SUP> in 550 iterations with less aggressive training, or PSNR's above 60 (i.e. MSE below 10<SUP>-6</SUP>) in 100 iterations with aggressive training where SIREN requires around 10 000 iterations to reach a MSE of around 10<SUP>-5</SUP>, corresponding to a PSNR of 50.

Below is a plot of the PSNR's achieved in a less aggressive training run and a video of the same training run:

![Plot of PSNR's.](https://raw.githubusercontent.com/mlaang/Neural-Networks--All-of-Whose-Derivatives-are-Self-Normalizing-Neural-Networks/master/PSNR-with-less-aggressive-training.png)

[![Video of images from fitting process.](https://img.youtube.com/vi/XYz6ayaKG_g/0.jpg)](https://www.youtube.com/watch?v=XYz6ayaKG_g)

Below is a plot of the PSNR's achieved in a aggressive training run:

![Plot of PSNR's.](https://raw.githubusercontent.com/mlaang/Neural-Networks--All-of-Whose-Derivatives-are-Self-Normalizing-Neural-Networks/master/PSNR-with-aggressive-training.png)

A write up explaining the mathematics and giving an overview can be found ![here](https://raw.githubusercontent.com/mlaang/Neural-Networks--All-of-Whose-Derivatives-are-Self-Normalizing-Neural-Networks/master/NN's%20All%20of%20Whose%20Derivatives%20are%20Self-Normalizing%20Neural%20Networks.pdf).
