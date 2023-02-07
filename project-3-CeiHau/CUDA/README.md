# Project 3 CUDA Version
## compile:
``` compile: nvcc cuda_version.cu -o cuda_version -arch=sm_61 ```

## run: 
``` ./cuda_version n  ``` <br><br>
For example, the following command means 1000 × 1000 grid resolution. <br>
``` ./cuda_version 1000 ``` <br>

I hard coded the NBLOCKS and NTHREADS_PER_BLOCK. The number of Rays = NBLOCKS * NTHREADS_PER_BLOCK

## generate the image:
I used python to generate the image in ```image_script.ipynb```.
