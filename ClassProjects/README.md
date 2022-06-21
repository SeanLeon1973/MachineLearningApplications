# Deep Convolutional Generative Adveserial Network (DCGAN.py)

The code within DCGAN.py is inspired by the work of (https://github.com/jayleicn/animeGAN) and the gradient penalty discussed in (https://arxiv.org/abs/1705.07215). The dataset utilized was sourced from the "I'm Something of a Painter Myself" Kaggle project (https://www.kaggle.com/code/greatrxt/monet/data). This python file generates unique Monet images from random noise to the best of its ability. This was done through a GAN's discriminator and generator architecture. Some example output images are shown below.

# Deep Regret Analytic Generative Adversarial Network (DRAGAN.py)

The code within DRAGAN.py is based on (https://arxiv.org/abs/1705.07215), which is the source of the gradient penalty in DCGAN.py. This approach utilized the same data as DCGAN.py. This GAN is ill-suited for image-based learning problems because of the architecture's large number of linear layers. As a result, the resulting generated images are of low quality. The resulting images are shown below. 

# Monetify.py

This is based on the work in DCGAN.py; however, the input to the generator is not a vector of random noise but rather a stock image. I found crisp, defined shapes to be challenging for DCGAN.py to generate. As a result, I gave it a stock image and looked to see if the DCGAN could morph that image into a Monet-like image. The resulting images are shown below. 