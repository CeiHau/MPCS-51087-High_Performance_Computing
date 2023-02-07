# Milestone1: Serial Version

## compile:
``` g++ -O3 serial.cpp -o serial ```

## run: 
``` ./serial n N_rays ``` <br><br>
For example, the following command means 1000 × 1000 grid resolution and 10000000 rays. <br>
``` ./serial 1000 10000000 ```

## generate the image:
I used python to generate the image in ```image_script.ipynb```.

With the arguments n = 1000, N_rays = 10000000. I need to multiplied the result by 255 to produce the image. The multiplier may change with the different arguments.

