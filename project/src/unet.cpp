/************************************************************************************
***
*** Copyright 2024 Dell(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Sat 24 Aug 2024 03:22:25 PM CST
***
************************************************************************************/

#include "unet.h"

TENSOR *unet_forward(UNetModel *unet,
	TENSOR *image_latent, TENSOR *timesteps, TENSOR *cond_latent, TENSOR *cond_pooled, 
    std::vector<TENSOR *>controls, float control_strength)
{
    TENSOR *argv[] = {image_latent, timesteps, cond_latent, cond_pooled};
    // unet->controls = controls;
    unet->control_strength = control_strength;
    TENSOR *cond_output = unet->engine_forward(ARRAY_SIZE(argv), argv);

    return cond_output;
}
