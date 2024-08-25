/************************************************************************************
***
*** Copyright 2024 Dell(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Sat 24 Aug 2024 03:22:25 PM CST
***
************************************************************************************/

#include "sdxl.h"

#include "vae.h"
#include "clip.h"
#include "unet.h"
#include <ggml_engine.h>

struct ControlNet {
    int a = 100; // dummy
};

typedef std::function<TENSOR *(TENSOR*, float, int)> denoiser_cb_t;
void k_sample(denoiser_cb_t denoise_model, TENSOR *x, std::vector<float> sigmas, RNG* rng);
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
    RNG *rng, Denoiser *denoiser);
// ----------------------------------------------------------------------------------------------------------------------------

void k_sample(denoiser_cb_t denoise_model, TENSOR *x, std::vector<float> sigmas, RNG* rng)
{
    sample_method_t method = EULER_A;
    int n = x->batch * x->chan * x->height * x->width;

    size_t steps = sigmas.size() - 1;
    // sample_euler_ancestral
    switch (method) {
    case EULER_A: {
        TENSOR *noise = tensor_copy(x);
        TENSOR *d = tensor_copy(x);

        for (int i = 0; i < steps; i++) {
            float sigma = sigmas[i];

            // denoise
            TENSOR *noised_output = denoise_model(x, sigma, i + 1); // unet->forward ...
            check_avoid(noised_output);

            // d = (x - noised_output) / sigma
            {
                for (int j = 0; j < n; j++) {
                    d->data[j] = (x->data[j] - noised_output->data[j]) / sigma; // d = (x - noised_output)/sigma
                }
            }

            // get_ancestral_step
            float sigma_up = std::min(sigmas[i + 1],
                std::sqrt(sigmas[i + 1] * sigmas[i + 1] * (sigmas[i] * sigmas[i] - sigmas[i + 1] * sigmas[i + 1]) / (sigmas[i] * sigmas[i])));
            float sigma_down = std::sqrt(sigmas[i + 1] * sigmas[i + 1] - sigma_up * sigma_up);

            // Euler method
            float dt = sigma_down - sigmas[i];
            // x = x + d * dt
            {
                for (int j = 0; j < n; j++) {
                    x->data[j] = x->data[j] + d->data[j] * dt;
                }
            }

            if (sigmas[i + 1] > 0) { // add noise
                set_scale_randn(noise, rng, sigma_up);
                for (int j = 0; j < n; j++) {
                    x->data[j] = x->data[j] + noise->data[j]; // x = x + sgigma_up * noise
                }
            }
            tensor_destroy(noised_output);
        } // end of for
        tensor_destroy(d);
        tensor_destroy(noise);
    } break;
    case EULER: // Implemented without any sigma churn
    {
        TENSOR *d = tensor_copy(x);; // d = 0

        for (int i = 0; i < steps; i++) {
            float sigma = sigmas[i];
            // denoise
            TENSOR *noised_output = denoise_model(x, sigma, i + 1);
            // d = (x - noised_output) / sigma
            {
                for (int j = 0; j < n; j++) {
                    d->data[j] = (x->data[j] - noised_output->data[j]) / sigma; // d = (x - noised_output)/sigma
                }
            }

            float dt = sigmas[i + 1] - sigma;
            // x = x + d * dt
            {
                for (int j = 0; j < n; j++) {
                    x->data[j] = x->data[j] + d->data[j] * dt; // x = x + dt * d
                }
            }
            tensor_destroy(noised_output);
        } // end of for
        tensor_destroy(d);
    } break;

    default:
        syslog_error("Attempting to sample with nonexisting sample method %i", method);
        abort();
    }
}


struct GGMLModel *load_model(ModelConfig *config)
{
    struct GGMLModel *model = new GGMLModel();
    model->preload(config->model_path);
    model->remap("first_stage_model.", "vae.");
    model->remap("model.diffusion_model.", "unet.");
    model->remap(".transformer_blocks.", ".transformer.");
    model->remap("cond_stage_model.transformer.text_model.", "clip.text_model.");
    model->remap("cond_stage_model.1.transformer.text_model.", "clip.text_model2.");

    return model;
}

void config_dump(ModelConfig *config)
{
    printf("Input: \n");
    printf("    device:            %d\n", config->device);
    printf("    model path:        '%s'\n", config->model_path);
    printf("    input image:       '%s'\n", config->input_image);
    printf("    prompt:            '%s'\n", config->positive);
    printf("    negative prompt:   '%s'\n", config->negative);
    printf("    noise strength:    %.2f\n", config->noise_strength);
    printf("    sample steps:      %d\n", config->sample_steps);
    printf("    seed:              %d\n", config->seed);
    printf("Control:\n");
    printf("    control model:     '%s'\n", config->control_model_path);
    printf("    control image:     '%s'\n", config->control_image_path);
    printf("    control strength:  %.2f\n", config->control_strength);
    printf("Output:\n");
    printf("    output height:     %d\n", config->height);
    printf("    output width:      %d\n", config->width);
    printf("    output path:       '%s'\n", config->output_path);
}

void config_init(ModelConfig *config)
{
    if (config->seed < 0) {
        srand((int)time(NULL));
        config->seed = rand();

        config->rng.manual_seed(config->seed);
        config->denoiser.init();
    }

    config->height -= config->height % 64;
    config->width -= config->width % 64;

    // init sigmas
    std::vector<float> sigmas = config->denoiser.get_sigmas(config->sample_steps);
    if (strlen(config->input_image) < 1) {
        config->sigmas = sigmas;
    } else {
        size_t t_enc = static_cast<size_t>(config->sample_steps * config->noise_strength);
        if (t_enc >= (size_t)config->sample_steps)
            t_enc = (size_t)config->sample_steps - 1;

        std::vector<float> sigma_sched;
        sigma_sched.assign(sigmas.begin() + config->sample_steps - t_enc - 1, sigmas.end());

        config->sigmas = sigma_sched;
    }

    if (config->verbose)
        config_dump(config);
}


// x = image_latent + sigmas[0]*noise
TENSOR *config_noised_latent(ModelConfig *config, TENSOR *image_latent)
{
    int B = 1;
    int C = 4;
    int H = config->height / 8;
    int W = config->width / 8;

    TENSOR *x = tensor_create(B, C, H, W);
    CHECK_TENSOR(x);

    set_scale_randn(x, &(config->rng), config->sigmas[0]);
    if (image_latent != NULL) {
        for (int j = 0; j < B * C * H * W; j++) {
            x->data[j] += image_latent->data[j];
        }
    }

    return x;
}



int image2image(ModelConfig *config)
{
    Denoiser denoiser;
    TextEncoder clip;
    AutoEncoderKL vae;
    UNetModel unet;
     

    printf("Creating image from image ...\n");

    config_init(config);

    GGMLModel *model = load_model(config);
    check_point(model != NULL);

    // model->dump();

    denoiser.init();

    clip.set_device(config->device);
    clip.start_engine();
    clip.load_weight(model, "clip.");

    std::vector<TENSOR *> positive_latent_pooled = clip_encode(&clip, config->positive, config->height, config->width);
    check_point(positive_latent_pooled.size() == 2);
    TENSOR *positive_latent = positive_latent_pooled[0];
    TENSOR *positive_pooled = positive_latent_pooled[1];
    check_point(positive_latent);
    check_point(positive_pooled);
    tensor_show((char *)"positive_latent", positive_latent);
    tensor_show((char *)"positive_pooled", positive_pooled);


    std::vector<TENSOR *> negative_latent_pooled = clip_encode(&clip, config->negative, config->height, config->width);
    check_point(negative_latent_pooled.size() == 2);
    TENSOR *negative_latent = negative_latent_pooled[0];
    TENSOR *negative_pooled = negative_latent_pooled[1];
    check_point(negative_latent);
    check_point(negative_pooled);
    tensor_show((char *)"negative_latent", negative_latent);
    tensor_show((char *)"negative_pooled", negative_pooled);

    clip.stop_engine();
    CheckPoint("OK !");




    vae.set_device(config->device);
    vae.start_engine();
    vae.load_weight(model, "vae.");

    TENSOR *image_tensor = tensor_load_image(config->input_image, 0 /*with alpha */);
    check_point(image_tensor);
    tensor_show("image_tensor", image_tensor);

    TENSOR *image_latent = vae_encode(&vae, image_tensor);
    check_tensor(image_latent);
    tensor_show("image_latent", image_latent);
    tensor_destroy(image_tensor);

    TENSOR *noised_image_latent = config_noised_latent(config, image_latent);
    CHECK_TENSOR(noised_image_latent);
    tensor_show("noised_image_latent", noised_image_latent);

    // -----------------------------------------------------------------------------------------
    unet.set_device(config->device);
    unet.start_engine();
    unet.load_weight(model, "unet.");

    // TENSOR *one_batch_sample(
    //     UNetModel *unet,
    //     TENSOR* x,
    //     TENSOR* positive_latent,
    //     TENSOR* positive_pooled,
    //     TENSOR* negative_latent,
    //     TENSOR* negative_pooled,
    //     float config_scale,
    //     ControlNet *control_net,
    //     TENSOR* control_image, // like canny image ...
    //     float control_strength,
    //     const std::vector<float>& sigmas,
    //     RNG *rng, Denoiser *denoiser)
    TENSOR *s0 = one_batch_sample(&unet, 
        image_latent, positive_latent, positive_pooled, negative_latent, negative_pooled, config->config_scale,
        NULL /*contrl_net */, 
        NULL /*control_image */,
        config->control_strength,
        config->sigmas,
        &(config->rng),
        &(config->denoiser)
        );

    unet.stop_engine();
    CheckPoint("OK !");
    // -----------------------------------------------------------------------------------------

    TENSOR *y = vae_decode(&vae, s0); // image_latent);
    check_point(y);
    tensor_show("y", y);

    tensor_saveas_image(y, 0, "/tmp/y.png");

    vae.stop_engine();
    tensor_destroy(y);
    tensor_destroy(image_latent);

    CheckPoint("OK !");

    model->clear();

    CheckPoint("OK !");

    return 0;
}

int text2image(ModelConfig *config)
{
    printf("Creating image from text ...\n");

    config_dump(config);

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

    CheckPoint("steps: %ld", steps);

    TENSOR* noised_input = tensor_copy(x); // place holder for denoise_model ...
    CHECK_POINT(noised_input);
    tensor_show("----------------- place holder noised_input", noised_input);

    TENSOR* noised_output = tensor_copy(x);  // place holder for denoise_model ...
    CHECK_TENSOR(noised_output);
    tensor_show("----------------- place holder noised_output", noised_output);

    CheckPoint("-----------------------");

    auto denoise_model = [&](TENSOR* input, float sigma, int step) -> TENSOR * {
        // f32 [   128,   128,     4,     1], input
        int ne_elements = input->batch * input->chan * input->height * input->width;

        CheckPoint("sigma = %.4f, step = %d, ne_elements = %d", sigma, step, ne_elements); 

        float c_skip = 1.0f;
        float c_out = 1.0f;
        float c_in = 1.0f;
        std::vector<float> scaling = denoiser->get_scalings(sigma);
        c_out = scaling[0];
        c_in  = scaling[1];

        TENSOR *timesteps = tensor_create(1, 1, 1, 1);
        CHECK_POINT(timesteps);
        timesteps->data[0] = denoiser->sigma_to_t(sigma); // denoiser ???
        CheckPoint("-----------------------------");
        tensor_show("----------------- timesteps", timesteps);

        // noised_input = input * c_in
        for (int i = 0; i < ne_elements; i++) {
            noised_input->data[i] = input->data[i] * c_in;
        }
        // timesteps -- c_out = -0.0292, c_in = 0.9996
        // f32 [     1,     1,     1,     1], timesteps
        // f32 [   128,   128,     4,     1], noised_input

        CheckPoint("-----------------------------");
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
        // -----------------------------------------------------------------------------------------------------------------
        // unet->controls = controls;
        // unet->control_strength = control_strength;
        // TENSOR *argv1[] = {noised_input, timesteps, positive_latent, positive_pooled};
        // TENSOR *positive_output = unet->engine_forward(ARRAY_SIZE(argv1), argv1);
        CheckPoint("-----------------------------");
        TENSOR *positive_output = unet_forward(unet, noised_input, timesteps, positive_latent, positive_pooled, controls, control_strength);
        CHECK_TENSOR(positive_output);
        // -----------------------------------------------------------------------------------------------------------------
        CheckPoint("-----------------------------");

        // uncond
        if (control_net != NULL && control_image != NULL) {
            TENSOR *argv[] = {noised_input, control_image, timesteps, negative_latent, negative_pooled};
            // TENSOR *temp = control_net->engine_forward(ARRAY_SIZE(argv), argv);
            // // controls = control_net->controls;
            // tensor_destroy(temp);
        }

        CheckPoint("-----------------------------");
        // unet->forward ...
        // -----------------------------------------------------------------------------------------------------------------
        // TENSOR *argv2[] = {noised_input, timesteps, negative_latent, negative_pooled};
        // unet->controls = controls;
        // unet->control_strength = control_strength;
        // TENSOR *negative_output = unet->engine_forward(ARRAY_SIZE(argv2), argv2);
        TENSOR *negative_output = unet_forward(unet, noised_input, timesteps, negative_latent, negative_pooled, controls, control_strength);
        CHECK_TENSOR(negative_output);
        // -----------------------------------------------------------------------------------------------------------------

        CheckPoint("--------------------------");
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
        CheckPoint("--------------------------");

        // xxx_debug
        tensor_destroy(negative_output);
        tensor_destroy(positive_output);
        tensor_destroy(timesteps);

        CheckPoint("--------------------------");
        return noised_output;
    };

    CheckPoint("--------------------------");

    k_sample(denoise_model, x, sigmas, rng); // for (int i = 0; i < steps; i++) ==> update x !!!
    CheckPoint("--------------------------");

    // tensor_destroy(noised_output); // noised_output has been destroy by k_sample ...
    tensor_destroy(noised_input);

    return x;
}

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>
#define GGML_NN_IMPLEMENTATION
#include <ggml_nn.h>

