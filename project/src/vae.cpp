/************************************************************************************
***
*** Copyright 2024 Dell(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Sat 24 Aug 2024 03:22:25 PM CST
***
************************************************************************************/

#include "vae.h"

// input -- big tensor, input_tile -- small tensor
static int get_input_tile(TENSOR *input, int start_row, int start_col, TENSOR *input_tile);
static float tile_smooth_step(const float x);
// output -- big tensor, output_tile -- small tensor
static int set_output_tile(TENSOR *output_tile, TENSOR *output, int start_row, int start_col, int overlap);


TENSOR *vae_encode(AutoEncoderKL *vae, TENSOR *image)
{
    float scale_factor = 0.130250; // VERSION_XL
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
    float scale_factor = 0.130250; // VERSION_XL 
    CHECK_TENSOR(latent);

    int n = latent->batch * latent->chan * latent->height * latent->width;
    for (int i = 0; i < n; i++) {
        latent->data[i] = latent->data[i]/scale_factor;
    }

    if (latent->height <= 96 && latent->width <= 96) { // same as raw image size < 768x768
        vae->encode_flag = false;
        TENSOR* argv[] = { latent };
        TENSOR* image = vae->engine_forward(ARRAY_SIZE(argv), argv); // output_tile
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

    // big latent ... 
    int tile_size = 96; // 32;
    int tile_overlap = 16;
    int non_tile_overlap = tile_size - tile_overlap;

    TENSOR *input_tile = tensor_create(latent->batch, 4 /*channel*/, tile_size, tile_size);
    CHECK_TENSOR(input_tile);

    TENSOR *output = tensor_create(latent->batch, 3/*channel*/, 8*latent->height, 8*latent->width);
    CHECK_TENSOR(output);

    bool last_row = false;
    for (int start_row = 0; start_row < latent->height && (! last_row); start_row += non_tile_overlap) {
        if (start_row + tile_size >= latent->height) {
            start_row = latent->height - tile_size;
            last_row = true;
        }

        // i in [start_row, start_row + tile_size)
        bool last_col = false;
        for (int start_col = 0; start_col < latent->width && (! last_col); start_col += non_tile_overlap) {
            syslog_info("vae decode at (%d, %d) with tile size %d ...", start_row, start_col, tile_size);

            if (start_col + tile_size >= latent->width) {
                start_col = latent->width - tile_size;
                last_col = true;
            }
            // j in [start_col, start_col + tile_size]
            CHECK_POINT(get_input_tile(latent, start_row, start_col, input_tile) == RET_OK);

            vae->encode_flag = false;
            TENSOR* argv[] = { input_tile };
            TENSOR* output_tile = vae->engine_forward(ARRAY_SIZE(argv), argv); // output_tile
            CHECK_TENSOR(output_tile);
            CHECK_POINT(set_output_tile(output_tile, output, 8*start_row, 8*start_col, 8*tile_overlap) == RET_OK);
            tensor_destroy(output_tile);
        }
    }
    tensor_destroy(input_tile);

    // output_tile from [-1.0, 1.0] to [0.0, 1.0]
    {
        n = output->batch * output->chan * output->height * output->width;
        for (int i = 0; i < n; i++) {
            output->data[i] = CLAMP((output->data[i] + 1.0)/2.0, 0.0, 1.0);
        }
    }

    return output;
}


// input -- big tensor, input_tile -- small tensor
static int get_input_tile(TENSOR *input, int start_row, int start_col, TENSOR *input_tile)
{
    check_tensor(input);
    check_tensor(input_tile);
    check_point(input->batch == input_tile->batch && input->chan == input_tile->chan);
    check_point(input_tile->height == input_tile->width); // suppose input_tile is squre ...
    int tile_size = input_tile->height;

    check_point(start_row >= 0 && start_row + tile_size <= input->height);
    check_point(start_col >= 0 && start_col + tile_size <= input->width);

    for (int b = 0; b < input->batch; b++) {
        for (int c = 0; c < input->chan; c++) {
            for (int i = 0; i < tile_size; i++) {
                float *src = tensor_start_row(input, b, c, start_row + i);
                float *dst = tensor_start_row(input_tile, b, c, i);
                for (int j = 0; j < tile_size; j++) {
                    dst[j] = src[start_col + j];
                }
            }
        }
    }
    return RET_OK;
}

static float tile_smooth_step(const float x)
{
    GGML_ASSERT(x >= 0.f && x <= 1.f);
    // x = 0.0 ==>  return 0.0
    // x = 0.5 ==>  return 0.5
    // x = 1.0 ==>  return 1.0
    return x * x * x * (x * (6.0f * x - 15.0f) + 10.0f);
}

// output -- big tensor, output_tile -- small tensor
static int set_output_tile(TENSOR *output_tile, TENSOR *output, int start_row, int start_col, int overlap)
{
    check_tensor(output_tile);
    check_tensor(output);

    check_point(output->batch == output_tile->batch && output->chan == output_tile->chan);
    check_point(output_tile->height == output_tile->width); // suppose output_tile is squre ...
    int tile_size = output_tile->height;
    check_point(start_row >= 0 && start_row + tile_size <= output->height);
    check_point(start_col >= 0 && start_col + tile_size <= output->width);

    for (int b = 0; b < output->batch; b++) {
        for (int c = 0; c < output->chan; c++) {
            for (int i = 0; i < tile_size; i++) {
                float *src = tensor_start_row(output_tile, b, c, i);
                float *dst = tensor_start_row(output, b, c, start_row + i);

                const float i_f_0 = (start_row > 0) ? float(i) / overlap : 1.0;
                const float i_f_1 = (start_row < (output->height - tile_size)) ? float(tile_size - i) / float(overlap) : 1.0;
                const float i_f = std::min(std::min(i_f_0, i_f_1), 1.0f);

                for (int j = 0; j < tile_size; j++) {
                    const float j_f_0 = (start_col > 0) ? float(j) / float(overlap) : 1.0;
                    const float j_f_1 = (start_col < (output->width - tile_size)) ? float(tile_size - j)/float(overlap) : 1.0 ;
                    const float j_f = MIN(MIN(j_f_0, j_f_1), 1.0);

                    float new_value = src[j];
                    float old_value = dst[start_col + j];
                    dst[start_col + j] = old_value + new_value * tile_smooth_step(i_f) * tile_smooth_step(j_f);
                }
            }
        }
    }
    return RET_OK;
}
