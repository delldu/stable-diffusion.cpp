 #/************************************************************************************
#***
#***	Copyright 2024 Dell(18588220928@163.com), All Rights Reserved.
#***
#***	File Author: Dell, Wed 24 Apr 2024 03:51:18 PM CST
#***
#************************************************************************************/
#
#! /bin/sh

PROMPT="a children bag"
NEGATIVE="black and white"
IMAGE=images/bag.png
NOISE_STRENGTH=0.80
CONTROL_STRENGTH=0.70
SD_MODELS_DIR=~/WDisk/Workspace/2023-07-01/sd_models

usage()
{
	echo "Usage: $0 [options]"
	echo "Options:"

	echo "-----------------------------------------------------"
	echo "convert_sd21_turbo"
	echo "    Convert sd21 turbo model"
	echo "convert_sdxl_turbo"
	echo "    Convert sdxl tubrbo model"
	echo
	echo "-----------------------------------------------------"
	echo "test_sd21_img2img --image <image.png> --promt <s1> --negative <s2> --noise <$NOISE_STRENGTH> --control <$CONTROL_STRENGTH>"
	echo "    Test SD21 image to image"
	echo "test_sdxl_img2img --image <image.png> --promt <s1> --negative <s2> --noise <$NOISE_STRENGTH> --control <$CONTROL_STRENGTH>"
	echo "    Test SDXL image to image"
	
	exit 1
}

convert_sd21_turbo()
{
	mkdir -p models

	# https://huggingface.co/stabilityai/sd-turbo
	# Step 1) convert base model
	# sd-turbo comes from sd21, no need update its vae ...
	./build/bin/sd --mode convert --model ${SD_MODELS_DIR}/sd-turbo/sd_turbo.safetensors -o models/sd21_turbo_q8_0.gguf --type q8_0

	# Step 2) convert canny model ...
	./build/bin/sd --mode convert --model ${SD_MODELS_DIR}/sd2.1/control_v11p_sd21_canny.safetensors -o models/sd21_canny_q8_0.gguf --type q8_0
	./build/bin/sd --mode convert --model ${SD_MODELS_DIR}/sd2.1/control_v11p_sd21_hed.safetensors -o models/sd21_hed_q8_0.gguf --type q8_0
	./build/bin/sd --mode convert --model ${SD_MODELS_DIR}/sd2.1/control_v11p_sd21_lineart.safetensors -o models/sd21_lineart_q8_0.gguf --type q8_0
	./build/bin/sd --mode convert --model ${SD_MODELS_DIR}/sd2.1/control_v11p_sd21_color.safetensors -o models/sd21_color_q8_0.gguf --type q8_0
	./build/bin/sd --mode convert --model ${SD_MODELS_DIR}/sd2.1/control_v11p_sd21_ade20k.safetensors -o models/sd21_ade20k_q8_0.gguf --type q8_0
	./build/bin/sd --mode convert --model ${SD_MODELS_DIR}/sd2.1/control_v11p_sd21_depth.safetensors -o models/sd21_depth_q8_0.gguf --type q8_0
	./build/bin/sd --mode convert --model ${SD_MODELS_DIR}/sd2.1/control_v11p_sd21_normalbae.safetensors -o models/sd21_normal_q8_0.gguf --type q8_0
	./build/bin/sd --mode convert --model ${SD_MODELS_DIR}/sd2.1/control_v11p_sd21_scribble.safetensors -o models/sd21_scribble_q8_0.gguf --type q8_0
	./build/bin/sd --mode convert --model ${SD_MODELS_DIR}/sd2.1/control_v11p_sd21_openpose.safetensors -o models/sd21_openpose_q8_0.gguf --type q8_0
}

convert_sdxl_turbo()
{
	mkdir -p models

	# Step 1) convert base model, need update vae_fp16
	./build/bin/sd --mode convert \
		--model ${SD_MODELS_DIR}/sd_xl_turbo/sd_xl_turbo_1.0_fp16.safetensors \
		--vae ${SD_MODELS_DIR}/sdxl1.0/sdxl_vae_fp16_fix.safetensors \
		-o models/sdxl_turbo_q8_0.gguf --type q8_0

	# Step 2) convert canny model
	# python project/convert_ctrlnet.py \
	# 	--unet_input ../sd_models/sd_xl_turbo/sd_xl_turbo_1.0_fp16.safetensors \
	# 	--ctrl_input ../sd_models/sdxl1.0/control-lora-canny-rank128.safetensors \
	# 	--ctrl_output /tmp/ctrl_output.pth
	# ./build/bin/sd --mode convert --model /tmp/ctrl_output.pth -o models/sdxl_canny_q8_0.gguf -v --type q8_0
	# rm -rf /tmp/ctrl_output.pth
}


test_sd21_img2img()
{
	mkdir -p output

	i=0
	while [ $i -lt 2 ] ; 
	do
		case $1 in
		--pr*)
			PROMPT=$2
			shift
			shift
			;;
		--ne*)
			NEGATIVE=$2;
			shift
			shift
			;;
		--im*)
			IMAGE=$2;
			shift
			shift
			;;
		--no*)
			NOISE_STRENGTH=$2;
			shift
			shift
			;;
		--co*)
			CONTROL_STRENGTH=$2;
			shift
			shift
			;;
		esac
		i=`expr $i + 1`
	done	

	echo ./build/bin/sd --mode img2img \
		--model models/sd21_turbo_q8_0.gguf \
		--height 640 --width 512 \
	 	--init-image "$IMAGE" \
		--prompt "'$PROMPT'" \
		--negative-prompt "'$NEGATIVE'" \
		--strength "$NOISE_STRENGTH" \
		--control-net models/sd21_canny_q8_0.gguf \
		--control-image "$IMAGE" --canny \
		--control-strength "$CONTROL_STRENGTH" \
		--cfg-scale 7.5 \
		--steps 25 \
		-o output/img2img_sd21.png
}


test_sdxl_img2img()
{
	mkdir -p output

	i=0
	while [ $i -lt 2 ] ; 
	do
		case $1 in
		--pr*)
			PROMPT=$2
			shift
			shift
			;;
		--ne*)
			NEGATIVE=$2;
			shift
			shift
			;;
		--im*)
			IMAGE=$2;
			shift
			shift
			;;
		--no*)
			NOISE_STRENGTH=$2;
			shift
			shift
			;;
		--co*)
			CONTROL_STRENGTH=$2;
			shift
			shift
			;;
		esac
		i=`expr $i + 1`
	done

	echo ./build/bin/sd --mode img2img --model models/sdxl_turbo_q8_0.gguf \
		--height 1024 --width 800 \
	 	--init-image "$IMAGE" \
		--prompt "'$PROMPT'" \
		--negative-prompt "'$NEGATIVE'" \
		--strength "$NOISE_STRENGTH" \
		--cfg-scale 1.8 \
		--steps 5 \
		-o output/img2img_sdxl.png
}

if [ "$*" == "" ] ;
then
	usage
else
	eval "$*"
fi

# ./build/bin/sd --mode txt2img --model models/sd21_turbo_q8_0.gguf --prompt 'a children bag' --negative-prompt 'black and white' --strength 0.80 --control-net models/sd21_scribble_q8_0.gguf --control-image images/bag_scribble.png --control-strength 0.70 --cfg-scale 7.5 --steps 25 -o output/txt2img_sd21.png

