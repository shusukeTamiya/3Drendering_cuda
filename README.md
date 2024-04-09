# practical-work-2

## Image Processing by OPEN-OMP

The results will be saved in the folder named as "output".

Move to main directory:

```bash
cd OpenCV-OMP
```

### Exercise 2.1.1

The program requires two arguments to run correctly. The first argument is the path to the stereo image, and the second argument is the type of anaglyph to generate.

The anaglyph types are as follows:

- True_Anaglyphs
- Gray_Anaglyphs
- Color_Anaglyphs
- Half_Color_Anaglyphs
- Optimized_Anaglyphs

Usage:

```bash
./PW2_1_1 <image_path> <anaglyph_type>
```

Example:

```bash
g++ PW2_1_1.cpp -fopenmp `pkg-config opencv4 --cflags` -c
g++ PW2_1_1.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o PW2_1_1
./PW2_1_1 test.jpg True_Anaglyphs
```

### Exercise 2.1.2

Usage:

```bash
./PW2_1_2 <image_path> <kernel_size> <sigma>
```

Example:

```bash
g++ PW2_1_2.cpp -fopenmp `pkg-config opencv4 --cflags` -c
g++ PW2_1_2.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o PW2_1_2
./PW2_1_2 test.jpg 5 0.5

```

### Exercise 2.1.3

Usage:

```bash
./PW2_1_3 <image_path> <neighborhood_size> <factor_ratio>
```

Example:

```bash
g++ PW2_1_3.cpp -fopenmp `pkg-config opencv4 --cflags` -c
g++ PW2_1_3.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o PW2_1_3
./PW2_1_3 test.jpg 5 0.5
```

## Image Processing by CUDA

The results will be saved in the folder named as "output".

Move to main directory:

```bash
cd OpenCV-CUDA
```

### Exercise 2.1.1(Cuda)

The program requires two arguments to run correctly. The first argument is the path to the stereo image, and the second argument is the type of anaglyph to generate.

The anaglyph types are as follows:

- True_Anaglyphs
- Gray_Anaglyphs
- Color_Anaglyphs
- Half_Color_Anaglyphs
- Optimized_Anaglyphs

Usage:

```bash
./PW2-1-1_cuda <image_path> <anaglyph_type>
```

Example:

```bash
/usr/local/cuda-11.6/bin/nvcc PW2-1-1_cuda.cu `pkg-config opencv4 --cflags --libs` PW2-1-1_cuda.cpp -o PW2-1-1_cuda
./PW2-1-1_cuda test.jpg True_Anaglyph
```

### Exercise 2.1.2(Cuda)

Usage:

./PW2-1-1_cuda <image_path> <kernel_size> <sigma>

Example:

```bash
/usr/local/cuda-11.6/bin/nvcc PW2-1-2_cuda.cu `pkg-config opencv4 --cflags --libs` PW2-1-2_cuda.cpp -o PW2-1-2_cuda
./PW2-1-2_cuda test.jpg 5 0.5

```

### Exercise 2.1.3(Cuda)

- Usage:

```bash
./PW2-1-3_cuda <image_path> <neighborhood_size> <factor_ratio>
```

Example:

```bash
/usr/local/cuda-11.6/bin/nvcc PW2-1-3_cuda.cu `pkg-config opencv4 --cflags --libs` PW2-2_cuda.cpp -o PW2-1-3_cuda
 ./PW2-1-3_cuda painting.tif 5 3
```

### Exercise 2.2(Cuda)

- Usage:

```bash
./PW2-2_cuda <image_path> <kernel_size> <sigma>
```

Example:

```bash
/usr/local/cuda-11.6/bin/nvcc PW2-2_cuda.cu `pkg-config opencv4 --cflags --libs` PW2-2_cuda.cpp -o PW2-2_cuda
 ./PW2-2_cuda painting.tif 5 3
```
