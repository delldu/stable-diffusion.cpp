## Download

https://huggingface.co/stabilityai/sd-turbo
https://huggingface.co/stabilityai/sdxl-turbo

SDXL-Turbo not support controlnet


## Build

mkdir build && cd build
cmake .. -DSD_CUBLAS=ON
cd ..

## Test
./project/b.sh

## Install
cp build/sd /usr/local/bin/sdxl_turbo
