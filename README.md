# Deep Learned Super Sampling
## [Try The Web App](https://vee-upatising.github.io/model.html)
Super Sampling Images to ```4``` Times Their Original Size using Convolutional Neural Networks <br/>



# Inspiration
This [Computerphile Video](https://www.youtube.com/watch?v=_DPRt3AcUEY) inpsired me to try to code a DLSS program <br/>
The video talks about super sampling from 1080p to 4K <br/>
Due to memory constraints, I can only work with smaller images

# Results
I compared [Nearest Neighbor](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize) Interpolation to my Deep Learned Super Sampling program <br/>
Here I am upsampling from ```128x128``` to ```256x256``` which is equivalent to increasing the size by ```16``` times

![flip](https://vee-upatising.github.io/images/flip.gif)
![anime](https://vee-upatising.github.io/images/sr.jpg)

# Model Architecture
![model](https://raw.githubusercontent.com/vee-upatising/DLSS/master/Results/model.png)

# Convolutional Filters Visualization
![conv](https://raw.githubusercontent.com/vee-upatising/DLSS/master/Results/conv.png)

# [Kaggle Notebook](https://www.kaggle.com/function9/deep-learned-super-sampling)
