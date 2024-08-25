/************************************************************************************
***
*** Copyright 2024 Dell(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Sat 24 Aug 2024 03:22:25 PM CST
***
************************************************************************************/

#include "denoiser.h"

void calculate_alphas_cumprod(float* alphas_cumprod)
{
    float start = 0.00085f;
    float end = 0.0120;

    float ls_sqrt = sqrtf(start);
    float le_sqrt = sqrtf(end);
    float amount = le_sqrt - ls_sqrt;
    float product = 1.0f;

    for (int i = 0; i < TIMESTEPS; i++) {
        float beta = ls_sqrt + amount * ((float)i / (TIMESTEPS - 1));
        product *= 1.0f - powf(beta, 2.0f);
        alphas_cumprod[i] = product;
    }
}

void set_scale_randn(TENSOR *t, RNG* rng, float scale)
{
    int n = t->batch * t->chan * t->height * t->width;
    std::vector<float> random_numbers = rng->randn(n);
    for (int i = 0; i < n; i++) {
        t->data[i] = random_numbers[i] * scale;
    }
}
