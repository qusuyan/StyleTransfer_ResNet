This program runs an improved version of real-time style transfer model propsed by Johnson et al. 

To run a demo:
Python main.py -c <path_to_content_image>

To train the model:
Python main.py -m 1 -s <path_to_style_image> -e <number_of_epoches>

Training will use the images stored in ./data, the trained model will be saved in ./model. 