# For Serial
compile:  ```g++ -O3 serial.cpp -o serial``` <br>
run:  ```./serial 75```. In this example, ```75``` is the value of n. <br>
 result will be saved in ```serial_result.csv```

# For parallel
before compile run:  ``` module load openmpi ``` <br>
compile: ``` mpic++ -O3 parallel.cpp -o parallel ``` <br>

 run:
  ``` sbatch -A mpcs51087 batch_project2 ``` <br>
 If you want to change the value of n, you can the ```2048``` in the following command in ```batch_project2``` to another value. <br>
 ```mpirun ./parallel 2048```<br>
 result will be saved in ```parallel_result.csv```

# plotting script
In file ```plotting_script.ipynb```
