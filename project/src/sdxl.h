/************************************************************************************
***
*** Copyright 2024 Dell(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Sat 24 Aug 2024 03:22:25 PM CST
***
************************************************************************************/

#ifndef __SDXL_H__
#define __SDXL_H__

#include "denoiser.h"
#include <vector>

struct ModelConfig {
    int device = 1; // 0 -- cpu, 1 -- cuda 0

    // Input ...
    char *model_path = (char *)"/opt/ai_models/sdxl_turbo_q8_0.gguf"; // sdxl_turbo_q8_0.gguf";
    
    char *input_image = (char *)""; 

    char *positive = (char *)"interior design of a luxurious master bedroom, gold and marble furniture, luxury, intricate, breathtaking";
    char *negative = (char *)"ugly, deformed, noisy, blurry, NSFW";
    float config_scale   = 1.8f; // CONST !!!
    float noise_strength = 0.75f;
    int sample_steps = 5;
    int seed = -1;

    // Control ...
    char *control_model_path = (char *)"/opt/ai_models/sdxl_lineart.gguf";
    char *control_image_path = (char *)"";
    float control_strength = 0.9f;

    // Output ...
    int width = 512;
    int height = 512;
    char *output_path = (char *)"output.png";

    int verbose = 1;

    // Program ...
    RNG rng;
    Denoiser denoiser;
    std::vector<float> sigmas;
};

void config_init(ModelConfig *config);
void config_dump(ModelConfig *config);

int text2image(ModelConfig *config);
int image2image(ModelConfig *config);

#endif // __SDXL_H__