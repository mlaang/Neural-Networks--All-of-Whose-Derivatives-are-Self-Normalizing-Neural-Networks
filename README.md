# Neural-Networks--All-of-Whose-Derivatives-are-Self-Normalizing-Neural-Networks

This is a project involving implicit representation learning using neural networks designed so that all their derivatives are self-normalizing neural networks. Another improvement which contributes to the result is a modification of the positional encoder introduced by Mildenhall et al. in their paper <I>NerRF: Representing Scenes as Neural Radiance Fields for View Synthesis</I>.

These methods substantially outperform methods like SIREN on the task of fitting images, being able to achieve PSNR's above 80, i.e. MSE below 10<SUP>-8</SUP> in 400 iterations with less aggressive training, or PSNR's above 60 (MSE below 10<SUP>-6</SUP>) in 100 iterations with aggressive training where SIREN requires around 10 000 iterations to reach a MSE of around 10<SUP>-5</SUP>, corresponding to a PSNR of 50.

<!----- Below is a plot of the PSNR's achieved in a training run and a video of the same training run: ----!>
