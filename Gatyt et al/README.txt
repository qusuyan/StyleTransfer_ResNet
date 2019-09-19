This is the algorithm proposed by Gatys et al. It can be used for style transfer and texture synthesis. 

Sample Usage for Style Transfer:
Python main.py -s <path_to_style_image> -c <path_to_content_image> -e <number_of_iterations>

Sample Usage for Texture Extraction:
Python main.py -m 1 -s <path_to_style_image>

By default, this program uses ./img/starry_night.jpg as style image and ./img/input.jpg as content image