/************************************************************************************
***
*** Copyright 2024 Dell(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Sat 24 Aug 2024 03:22:25 PM CST
***
************************************************************************************/

#include "vae.h"

TENSOR *vae_encode(AutoEncoderKL *vae, TENSOR *image)
{
    float scale_factor = 0.13025f; // VERSION_XL
    CHECK_TENSOR(image);

    TENSOR *latent = NULL;
    int n = image->batch * image->chan * image->height * image->width;

    // image from [0.0, 1.0] -- to [-1.0, 1.0]
    for (int i = 0; i < n; i++) {
        image->data[i] = (image->data[i] - 0.5) * 2.0;
    }

    vae->encode_flag = true;
    TENSOR* argv[] = { image };
    TENSOR *moment = vae->engine_forward(ARRAY_SIZE(argv), argv); // [1, 8, 128, 128]
    CHECK_TENSOR(moment);
    // add noise ?, scale_factor ...
    {
        latent = tensor_create(moment->batch, moment->chan/2, moment->height, moment->width);  // [1, 4, 128, 128]
        CHECK_TENSOR(latent);

        TENSOR *noise = tensor_create(moment->batch, moment->chan/2, moment->height, moment->width);
        CHECK_TENSOR(noise);
        {
            // #include <stdio.h>
            // #include <stdlib.h>
            // #include <time.h>
            srand((unsigned int)time(NULL));
            n = noise->batch * noise->chan * noise->height * noise->width;
            for (int i = 0; i < n; i++) {
                noise->data[i] = (float)rand() / (float)RAND_MAX;
            }
        }

        for (int b = 0; b < latent->batch; b++) {
            for (int c = 0; c < latent->chan; c++) {
                float *latent_data = tensor_start_chan(latent, b, c);
                float *moment_data = tensor_start_chan(moment, b, c);
                float *logvar_data = tensor_start_chan(moment, b, c + latent->chan); // moment->chan/2
                float *noise_data = tensor_start_chan(noise, b, c);

                n = latent->height * latent->width;
                for (int i = 0; i < n; i++) {
                    float mean = moment_data[i];
                    float logvar = CLAMP(logvar_data[i], -30.0, 20.0);
                    float std = std::exp(0.5f * logvar);
                    float value  = mean + std * noise_data[i];

                    latent_data[i] = value * scale_factor;
                }
            }
        }
        tensor_destroy(noise);
        tensor_destroy(moment);
    }

    return latent; // [1, 4, 128, 128]
}


TENSOR *vae_decode(AutoEncoderKL *vae, TENSOR *latent)
{
    float scale_factor = 0.13025f; // VERSION_XL 
    CHECK_TENSOR(latent);

    int n = latent->batch * latent->chan * latent->height * latent->width;
    for (int i = 0; i < n; i++) {
        latent->data[i] = latent->data[i]/scale_factor;
    }

    vae->encode_flag = false;
    TENSOR* argv[] = { latent };
    TENSOR* image = vae->engine_forward(ARRAY_SIZE(argv), argv);
    CHECK_TENSOR(image);
    // image from [-1.0, 1.0] to [0.0, 1.0]
    {
        n = image->batch * image->chan * image->height * image->width;
        for (int i = 0; i < n; i++) {
            image->data[i] = CLAMP((image->data[i] + 1.0)/2.0, 0.0, 1.0);
        }
    }

    return image;
}
