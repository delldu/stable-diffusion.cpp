/************************************************************************************
***
*** Copyright 2024 Dell(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Sat 24 Aug 2024 03:22:25 PM CST
***
************************************************************************************/

#include "sdxl.h"

#include <tensor.h>
#include "vae.h"
#include "clip.h"
#include "unet.h"
#include "denoiser.h"
#include <ggml_engine.h>

struct ControlNet {
    int a = 100; // dummy
};

struct GGMLModel *load_model(ModelConfig params)
{
    struct GGMLModel *model = new GGMLModel();
    model->preload(params.model_path);
    model->remap("first_stage_model.", "vae.");
    model->remap("model.diffusion_model.", "unet.");
    model->remap(".transformer_blocks.", ".transformer.");
    model->remap("cond_stage_model.transformer.text_model.", "clip.text_model.");
    model->remap("cond_stage_model.1.transformer.text_model.", "clip.text_model2.");

    return model;
}

void print_params(ModelConfig params)
{
    printf("Input: \n");
    printf("    device:            %d\n", params.device);
    printf("    model path:        '%s'\n", params.model_path);
    printf("    input image:       '%s'\n", params.input_image);
    printf("    prompt:            '%s'\n", params.positive);
    printf("    negative prompt:   '%s'\n", params.negative);
    printf("    noise strength:    %.2f\n", params.noise_strength);
    printf("    sample steps:      %d\n", params.sample_steps);
    printf("    seed:              %d\n", params.seed);
    printf("Control:\n");
    printf("    control model:     '%s'\n", params.control_model_path);
    printf("    control image:     '%s'\n", params.control_image_path);
    printf("    control strength:  %.2f\n", params.control_strength);
    printf("Output:\n");
    printf("    output height:     %d\n", params.height);
    printf("    output width:      %d\n", params.width);
    printf("    output path:       '%s'\n", params.output_path);
}

int text2image(ModelConfig params)
{
    Denoiser denoiser;
    TextEncoder clip;
    AutoEncoderKL vae;
    UNetModel unet;
     

    printf("Creating image from text ...\n");

    params.height -= params.height % 64;
    params.width -= params.width % 64;

    print_params(params);

    GGMLModel *model = load_model(params);
    check_point(model != NULL);

    // model->dump();

    denoiser.init();

    // clip.set_device(params.device);
    // clip.start_engine();
    // clip.load_weight(model, "clip.");

    // std::vector<TENSOR *> positive_latent_pooled = clip_encode(&clip, params.positive, params.height, params.width);
    // check_point(positive_latent_pooled.size() == 2);
    // TENSOR *positive_latent = positive_latent_pooled[0];
    // TENSOR *positive_pooled = positive_latent_pooled[1];
    // check_point(positive_latent);
    // check_point(positive_pooled);
    // tensor_show((char *)"positive_latent", positive_latent);
    // tensor_show((char *)"positive_pooled", positive_pooled);


    // std::vector<TENSOR *> negative_latent_pooled = clip_encode(&clip, params.negative, params.height, params.width);
    // check_point(negative_latent_pooled.size() == 2);
    // TENSOR *negative_latent = negative_latent_pooled[0];
    // TENSOR *negative_pooled = negative_latent_pooled[1];
    // check_point(negative_latent);
    // check_point(negative_pooled);
    // tensor_show((char *)"negative_latent", negative_latent);
    // tensor_show((char *)"negative_pooled", negative_pooled);

    // clip.stop_engine();
    // CheckPoint("OK !");


    vae.set_device(params.device);
    vae.start_engine();
    vae.load_weight(model, "vae.");

    TENSOR *x = tensor_load_image("../images/4guys.png", 0 /*with alpha */);
    check_point(x);
    tensor_show("x", x);

    TENSOR *z = vae_encode(&vae, x);
    check_tensor(z);
    tensor_show("z", z);

    TENSOR *y = vae_decode(&vae, z);
    check_point(y);
    tensor_show("y", y);

    tensor_saveas_image(y, 0, "/tmp/y.png");



    vae.stop_engine();

    CheckPoint("OK !");


    // unet.set_device(params.device);
    // unet.start_engine();
    // unet.load_weight(model, "unet.");

    // unet.stop_engine();
    // CheckPoint("OK !");


    // model->clear();




    // net.set_device(1);
    // net.start_engine();
    // // net.dump();

    // net.load_model(&model, "unet.");
    // // net.load_model(&model, "vae.");
    // // net.load_model(&model, "clip.");

    // net.stop_engine();


    // model.clear();

    // // Denoiser denoiser;
    // // denoiser.init();
    // // denoiser.dump();

    CheckPoint("OK !");

    return 0;
}

int image2image(ModelConfig params)
{
    printf("Creating image from image ...\n");

    params.height -= params.height % 64;
    params.width -= params.width % 64;

    print_params(params);

    return 0;
}

TENSOR *one_batch_sample(
    UNetModel *unet,
    TENSOR* x,
    TENSOR* positive_latent,
    TENSOR* positive_pooled,
    TENSOR* negative_latent,
    TENSOR* negative_pooled,
    float config_scale,
    ControlNet *control_net,
    TENSOR* control_image, // like canny image ...
    float control_strength,
    const std::vector<float>& sigmas,
    RNG *rng, Denoiser *denoiser)
{
    // for image to image
    // sample: control_strength = 0.9000, config_scale = 1.8000, sigmas.size() = 2
    // f32 [   128,   128,     4,     1],  x, hxw = 1024x1024
    // f32 [  2048,    77,     1,     1],  positive_latent
    // f32 [  2816,     1,     1,     1],  positive_pooled
    // f32 [  2048,    77,     1,     1],  negative_latent
    // f32 [  2816,     1,     1,     1],  negative_pooled
    // tensor == NULL // control_image

    // for text to image
    // sample: control_strength = 0.9000, config_scale = 1.8000, sigmas.size() = 6
    //    f32 [    64,    96,     4,     1], x, hxw = 768x512
    //    f32 [  2048,    77,     1,     1], positive_latent
    //    f32 [  2816,     1,     1,     1], positive_pooled
    //    f32 [  2048,    77,     1,     1], negative_latent
    //    f32 [  2816,     1,     1,     1], negative_pooled
    // tensor == NULL // control_image

    size_t steps = sigmas.size() - 1;
    TENSOR* noised_input = tensor_copy(x);
    CHECK_POINT(noised_input);
    TENSOR* noised_output = tensor_copy(x);
    CHECK_TENSOR(noised_output);

    auto denoise_model = [&](TENSOR* input, float sigma, int step) -> TENSOR * {
        // f32 [   128,   128,     4,     1], input
        int ne_elements = input->batch * input->chan * input->height * input->width;

        CheckPoint("sigma = %.4f, step = %d", sigma, step); 

        float c_skip = 1.0f;
        float c_out = 1.0f;
        float c_in = 1.0f;
        std::vector<float> scaling = denoiser->get_scalings(sigma);
        c_out = scaling[0];
        c_in  = scaling[1];

        TENSOR *timesteps = tensor_create(1, 1, 1, 1);
        CHECK_POINT(timesteps);
        timesteps->data[0] = denoiser->sigma_to_t(sigma); // denoiser ???

        // noised_input = input * c_in
        for (int i = 0; i < ne_elements; i++) {
            noised_input->data[i] = input->data[i] * c_in;
        }

        // timesteps -- c_out = -0.0292, c_in = 0.9996
        // f32 [     1,     1,     1,     1], timesteps
        // f32 [   128,   128,     4,     1], noised_input

        std::vector<TENSOR*> controls;
        // f32 [   128,   128,     4,     1], noised_input
        // f32 [     1,     1,     1,     1], timesteps
        // f32 [  2048,    77,     1,     1], positive_latent
        // f32 [  2816,     1,     1,     1], positive_pooled

        // cond
        if (control_net != NULL && control_image != NULL) {
            TENSOR *argv[] = {noised_input, control_image, timesteps, positive_latent, positive_pooled};
            // TENSOR *temp = control_net->engine_forward(ARRAY_SIZE(argv), argv);
            // // controls = control_net->controls;
            // tensor_destroy(temp);
        }
        // unet->forward ...
        TENSOR *argv1[] = {noised_input, timesteps, positive_latent, positive_pooled, NULL /*controls, control_strength*/};
        TENSOR *positive_output = unet->engine_forward(ARRAY_SIZE(argv1), argv1);

        // uncond
        if (control_net != NULL && control_image != NULL) {
            TENSOR *argv[] = {noised_input, control_image, timesteps, negative_latent, negative_pooled};
            // TENSOR *temp = control_net->engine_forward(ARRAY_SIZE(argv), argv);
            // // controls = control_net->controls;
            // tensor_destroy(temp);
        }
        // unet->forward ...
        TENSOR *argv2[] = {noised_input, timesteps, negative_latent, negative_pooled, NULL /*controls, control_strength */};
        TENSOR *negative_output = unet->engine_forward(ARRAY_SIZE(argv2), argv2);

        // update noised_output 
        {
            float latent_result;
            // config_scale -- 0.0 ==> negative
            // config_scale -- 0.5 ==> (positive + negativee)/2.0
            // config_scale -- 1.0 ==> positive
            // config_scale -- 2.0 ==> 2 * positive  - negative ...
            for (int i = 0; i < ne_elements; i++) {
                latent_result = negative_output->data[i] + config_scale * (positive_output->data[i] - negative_output->data[i]);
                noised_output->data[i] = latent_result * c_out + input->data[i] * c_skip; // c_skip == 1.0f
            }
        }

        tensor_destroy(negative_output);
        tensor_destroy(positive_output);
        tensor_destroy(timesteps);

        return noised_output;
    };

    k_sample(denoise_model, x, sigmas, rng); // for (int i = 0; i < steps; i++) ==> update x !!!
    // tensor_destroy(noised_output); // noised_output has been destroy by k_sample ...
    tensor_destroy(noised_input);

    return x;
}

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>
#define GGML_NN_IMPLEMENTATION
#include <ggml_nn.h>

