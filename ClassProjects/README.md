# Deep Convolutional Generative Adveserial Network (DCGAN.py)

The code within DCGAN.py is inspired by the work of (https://github.com/jayleicn/animeGAN) and the gradient penalty discussed in (https://arxiv.org/abs/1705.07215). The dataset utilized was sourced from the "I'm Something of a Painter Myself" Kaggle project (https://www.kaggle.com/code/greatrxt/monet/data). This python file generates unique Monet images from random noise to the best of its ability. This was done through a GAN's discriminator and generator architecture. Some example output images over several epochs are shown below.

<p float="center">
  <img src="https://user-images.githubusercontent.com/45838898/174730593-ee8dba2e-9d76-4efa-808d-e3992f07e754.jpg" width="250" />
  <img src="https://user-images.githubusercontent.com/45838898/174730610-17dc64f4-37fc-4c24-8dbd-78f2de16e740.jpg" width="250" /> 
  <img src="https://user-images.githubusercontent.com/45838898/174730647-5b71bc0c-8821-40ff-847f-5933cff8e91d.jpg" width="250" />
</p>

# Deep Regret Analytic Generative Adversarial Network (DRAGAN.py)

The code within DRAGAN.py is based on (https://arxiv.org/abs/1705.07215), which is the source of the gradient penalty in DCGAN.py. This approach utilized the same data as DCGAN.py. This GAN is ill-suited for image-based learning problems because of the architecture's large number of linear layers. As a result, the resulting generated images are of low quality. The resulting images are shown below. From right to left, the images are an output from the generator network and a reference image.

<p float="center">
  <img src="https://user-images.githubusercontent.com/45838898/174730807-942dff5e-c180-4d19-8188-4e9b53ec18db.jpg" width="375" />
  <img src="https://user-images.githubusercontent.com/45838898/174730845-27e20bd4-17c0-4b4e-bbc8-0f36b247fc52.jpg" width="375" />
</p>

# Monetify.py

This is based on the work in DCGAN.py; however, the input to the generator is not a vector of random noise but rather a stock image. I found crisp, defined shapes to be challenging for DCGAN.py to generate. As a result, I gave it a stock image and looked to see if the DCGAN could morph that image into a Monet-like image. The resulting images are shown below. 

<p float="center">
  <img src="https://user-images.githubusercontent.com/45838898/174731433-6c69e430-b531-422a-a3fd-1fce1dc6f2c1.png" width="375" />
  <img src="https://user-images.githubusercontent.com/45838898/174731376-236d8a05-d2d9-4a70-b15e-dca373cc3bff.png" width="375" /> 
</p>
