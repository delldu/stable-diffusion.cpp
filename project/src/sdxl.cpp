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
#include "adapter.h"

#include <ggml_engine.h>

static void euler_sample(
    UNetModel *unet,
    TENSOR* x, // image prompt
    TENSOR* positive_latent, TENSOR* positive_pooled, TENSOR* negative_latent, TENSOR* negative_pooled, float config_scale, // text prompt
    TENSOR* controls_0, TENSOR *controls_1, TENSOR *controls_2, TENSOR *controls_3,
    const std::vector<float>& sigmas,
    RNG *rng, Denoiser *denoiser);

static TENSOR *denoise_model(
    UNetModel *unet,
    TENSOR* positive_latent, TENSOR* positive_pooled, TENSOR* negative_latent, TENSOR* negative_pooled, float config_scale, // text prompt
    TENSOR* controls_0, TENSOR *controls_1, TENSOR *controls_2, TENSOR *controls_3,
    Denoiser *denoiser,
    TENSOR* input, float sigma, int step);

static TENSOR *config_latent(ModelConfig *config, TENSOR *image_latent);
static std::vector<TENSOR *>get_controls(ModelConfig *config);
static struct GGMLModel *load_model(ModelConfig *config);

// -----------------------------------------------------------------------------------------------------------------------------------------------
static struct GGMLModel *load_model(ModelConfig *config)
{
    struct GGMLModel *model = new GGMLModel();
    if (model != NULL && model->preload(config->model_path) == RET_OK) {
        model->preload(config->model_path);
        model->remap("first_stage_model.", "vae.");
        model->remap("model.diffusion_model.", "unet.");
        model->remap(".transformer_blocks.", ".transformer.");
        model->remap("cond_stage_model.transformer.text_model.", "clip.text_model.");
        model->remap("cond_stage_model.1.transformer.text_model.", "clip.text_model2.");

        return model;
    }

    if (model != NULL) {
        delete model;
    }

    return NULL;;
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
    }

    // config->seed &=0x7fffff; 
    config->rng.manual_seed(config->seed); // test case 1290286573, 1058759999, 1971437833
    config->denoiser.init();

    config->height -= config->height % 64;
    config->width -= config->width % 64;

    // init sigmas
    std::vector<float> sigmas = config->denoiser.get_sigmas(config->sample_steps);
    if (strlen(config->input_image) < 1) {
        config->sigmas = sigmas;
    } else {
        size_t t_enc = static_cast<size_t>(config->sample_steps * config->noise_strength); // noise strength ...
        if (t_enc >= (size_t)config->sample_steps)
            t_enc = (size_t)config->sample_steps - 1;

        std::vector<float> sigma_sched;
        sigma_sched.assign(sigmas.begin() + config->sample_steps - t_enc - 1, sigmas.end());

        config->sigmas = sigma_sched;
    }

    if (config->verbose)
        config_dump(config);
}

static std::vector<TENSOR *>get_controls(ModelConfig *config)
{
    std::vector<TENSOR *> res(4, NULL); // NEED free res[...] for get_output_tensor -- tensor_copy(...)

    if (strlen(config->control_image_path) < 1 || ! file_exist(config->control_image_path) 
        || ! file_exist(config->control_model_path) || config->control_strength < 0.01) {
        return res;
    }

    struct GGMLModel *model = new GGMLModel();
    if (model == NULL || model->preload(config->control_model_path) != RET_OK) {
        return res;
    }

    TENSOR *image_tensor = tensor_load_image(config->control_image_path, 0 /*alpha */); // like canny image 
    if (! tensor_valid(image_tensor)) {
        return res;
    }
    if (image_tensor->height != config->height || image_tensor->width != config->width) {
        TENSOR *t = image_tensor; // save as temp
        image_tensor = tensor_zoom(t, config->height, config->width);
        tensor_destroy(t);
        if (! tensor_valid(image_tensor))
            return res;
    }

    FullAdapterXL net;
    net.set_device(config->device);
    net.start_engine();
    net.load_weight(model, "adapter.");

    TENSOR *argv[] = { image_tensor };
    TENSOR *predict = net.engine_forward(ARRAY_SIZE(argv), argv);
    if (tensor_valid(predict)) { // useless
        tensor_destroy(predict);
    }

    // Save body0, body1, body2, body3 to res ...
    res[0] = net.get_output_tensor((char *)"body0");
    res[1] = net.get_output_tensor((char *)"body1");
    res[2] = net.get_output_tensor((char *)"body2");
    res[3] = net.get_output_tensor((char *)"body3");
    for (int i = 0; i < 4; i++) {
        if (! tensor_valid(res[i]))
            continue;

        int n = res[i]->batch * res[i]->chan * res[i]->height * res[i]->width;
        for (int j = 0; j < n; j++) {
            res[i]->data[j] = res[i]->data[j] * config->control_strength;
        }
    }

    // net.dump();
    net.stop_engine();

    if (model != NULL) {
        model->clear();
        delete model;
    }

    return res;
}



// x = image_latent + sigmas[0]*noise
static TENSOR *config_latent(ModelConfig *config, TENSOR *image_latent)
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
    TextEncoder clip;
    AutoEncoderKL vae;
    UNetModel unet;

    syslog_info("Creating image from image ...");
    config_init(config);

    TENSOR *image_tensor = tensor_load_image(config->input_image, 0 /*with alpha */);
    check_tensor(image_tensor);
    if (image_tensor->height != config->height || image_tensor->width != config->width) {
        TENSOR *t = image_tensor; // save as temp
        image_tensor = tensor_zoom(t, config->height, config->width);
        check_tensor(image_tensor);
        tensor_destroy(t);
    }

    GGMLModel *model = load_model(config);
    check_point(model != NULL);

    // -----------------------------------------------------------------------------------------
    clip.set_device(config->device);
    clip.start_engine();
    clip.load_weight(model, "clip.");

    std::vector<TENSOR *> positive_latent_pooled = clip_encode(&clip, config->positive, config->height, config->width);
    check_point(positive_latent_pooled.size() == 2);
    TENSOR *positive_latent = positive_latent_pooled[0];
    TENSOR *positive_pooled = positive_latent_pooled[1];
    check_tensor(positive_latent);
    check_tensor(positive_pooled);

    std::vector<TENSOR *> negative_latent_pooled = clip_encode(&clip, config->negative, config->height, config->width);
    check_point(negative_latent_pooled.size() == 2);
    TENSOR *negative_latent = negative_latent_pooled[0];
    TENSOR *negative_pooled = negative_latent_pooled[1];
    check_tensor(negative_latent);
    check_tensor(negative_pooled);

    clip.stop_engine();
    syslog_info("clip encode/decode OK.");

    // -----------------------------------------------------------------------------------------
    vae.set_device(config->device);
    vae.start_engine();
    vae.load_weight(model, "vae.");

    TENSOR *image_latent = vae_encode(&vae, image_tensor);
    check_tensor(image_latent);

    tensor_destroy(image_tensor);

    TENSOR *noised_latent = config_latent(config, image_latent);
    check_tensor(noised_latent);

    // -----------------------------------------------------------------------------------------
    {
        std::vector<TENSOR *> controls = get_controls(config);

        unet.set_device(config->device);
        unet.start_engine();
        unet.load_weight(model, "unet.");
        euler_sample(&unet, 
            noised_latent, positive_latent, positive_pooled, negative_latent, negative_pooled, config->config_scale,
            controls[0], controls[1], controls[2], controls[3],
            config->sigmas, &(config->rng), &(config->denoiser));
        unet.stop_engine();

        for (int i = 0; i < controls.size(); i++) {
            if (tensor_valid(controls[i]))
                tensor_destroy(controls[i]);
        }
        controls.clear();

        syslog_info("unet sample OK !");
    }
    // -----------------------------------------------------------------------------------------

    TENSOR *y = vae_decode(&vae, noised_latent);
    check_point(y);
    tensor_saveas_image(y, 0, config->output_path);

    vae.stop_engine();
    tensor_destroy(y);
    tensor_destroy(image_latent);
    syslog_info("vae decode OK !");

    model->clear();

    syslog_info("Creating image from image OK.");

    return RET_OK;
}

int text2image(ModelConfig *config)
{
    TextEncoder clip;
    AutoEncoderKL vae;
    UNetModel unet;

    syslog_info("Creating image from text ...");
    config_init(config);


    GGMLModel *model = load_model(config);
    check_point(model != NULL);

    // -----------------------------------------------------------------------------------------
    clip.set_device(config->device);
    clip.start_engine();
    clip.load_weight(model, "clip.");

    std::vector<TENSOR *> positive_latent_pooled = clip_encode(&clip, config->positive, config->height, config->width);
    check_point(positive_latent_pooled.size() == 2);
    TENSOR *positive_latent = positive_latent_pooled[0];
    TENSOR *positive_pooled = positive_latent_pooled[1];
    check_tensor(positive_latent);
    check_tensor(positive_pooled);

    std::vector<TENSOR *> negative_latent_pooled = clip_encode(&clip, config->negative, config->height, config->width);
    check_point(negative_latent_pooled.size() == 2);
    TENSOR *negative_latent = negative_latent_pooled[0];
    TENSOR *negative_pooled = negative_latent_pooled[1];
    check_tensor(negative_latent);
    check_tensor(negative_pooled);

    clip.stop_engine();
    syslog_info("clip encode/decode OK.");

    TENSOR *noised_latent = config_latent(config, NULL);
    check_tensor(noised_latent);

    // -----------------------------------------------------------------------------------------
    {
        std::vector<TENSOR *> controls = get_controls(config);

        unet.set_device(config->device);
        unet.start_engine();
        unet.load_weight(model, "unet.");

        euler_sample(&unet, 
            noised_latent, positive_latent, positive_pooled, negative_latent, negative_pooled, config->config_scale,
            controls[0], controls[1], controls[2], controls[3],
            config->sigmas, &(config->rng), &(config->denoiser));

        unet.stop_engine();

        for (int i = 0; i < controls.size(); i++) {
            if (tensor_valid(controls[i]))
                tensor_destroy(controls[i]);
        }
        controls.clear();

        syslog_info("unet sample OK.");
    }

    // -----------------------------------------------------------------------------------------
    vae.set_device(config->device);
    vae.start_engine();
    vae.load_weight(model, "vae.");

    TENSOR *y = vae_decode(&vae, noised_latent); 
    tensor_destroy(noised_latent);
    check_tensor(y);
    tensor_saveas_image(y, 0, config->output_path);
    tensor_destroy(y);

    vae.stop_engine();
    syslog_info("vae decode OK.");

    model->clear();

    syslog_info("Creating image from text OK.");

    return RET_OK;
}

static void euler_sample(
    UNetModel *unet,
    TENSOR* x, // image prompt
    TENSOR* positive_latent, TENSOR* positive_pooled, TENSOR* negative_latent, TENSOR* negative_pooled, float config_scale, // text prompt
    TENSOR* controls_0, TENSOR *controls_1, TENSOR *controls_2, TENSOR *controls_3,
    const std::vector<float>& sigmas,
    RNG *rng, Denoiser *denoiser)
{
    sample_method_t method = EULER_A;
    int n = x->batch * x->chan * x->height * x->width;

    size_t steps = sigmas.size() - 1;
    // sample_euler_ancestral
    switch (method) {
    case EULER_A: {
        TENSOR *d = tensor_copy(x);

        for (int i = 0; i < steps; i++) {
            syslog_info("Sample progress %d/%ld ...", i, steps);

            float sigma = sigmas[i];
            TENSOR *noised_output = denoise_model(
                unet,
                positive_latent, positive_pooled, negative_latent, negative_pooled, config_scale, // text prompt
                controls_0, controls_1, controls_2, controls_3,
                denoiser,
                x, sigma, i + 1 /* step */);
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
                TENSOR *noise = tensor_copy(x);
                check_avoid(noise);
                set_scale_randn(noise, rng, sigma_up);
                for (int j = 0; j < n; j++) {
                    x->data[j] = x->data[j] + noise->data[j]; // x = x + sgigma_up * noise
                }
                tensor_destroy(noise);
            }
            tensor_destroy(noised_output);
        } // end of for
        tensor_destroy(d);
    } break;
    case EULER: // Implemented without any sigma churn
    {
        TENSOR *d = tensor_copy(x);; // d = 0

        for (int i = 0; i < steps; i++) {
            syslog_info("Sample progress %d/%ld ...", i, steps);

            float sigma = sigmas[i];
            TENSOR *noised_output = denoise_model(
                unet,
                positive_latent, positive_pooled, negative_latent, negative_pooled, config_scale, // text prompt
                controls_0, controls_1, controls_2, controls_3,
                denoiser,
                x, sigma, i + 1 /* step */);
            check_avoid(noised_output);

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

static TENSOR *denoise_model(
    UNetModel *unet,
    TENSOR* positive_latent, TENSOR* positive_pooled, TENSOR* negative_latent, TENSOR* negative_pooled, float config_scale, // text prompt
    TENSOR* controls_0, TENSOR *controls_1, TENSOR *controls_2, TENSOR *controls_3,
    Denoiser *denoiser,
    TENSOR* input, float sigma, int step)
{
    // f32 [   128,   128,     4,     1], input
    int n = input->batch * input->chan * input->height * input->width;

    float c_skip = 1.0f;
    float c_out = 1.0f;
    float c_in = 1.0f;
    std::vector<float> scaling = denoiser->get_scalings(sigma);
    c_out = scaling[0];
    c_in  = scaling[1];

    TENSOR *noised_input = tensor_copy(input);
    CHECK_TENSOR(noised_input);
    // noised_input = input * c_in
    for (int i = 0; i < n; i++) {
        noised_input->data[i] = input->data[i] * c_in;
    }
    TENSOR *timesteps = tensor_create(1, 1, 1, 1);
    CHECK_TENSOR(timesteps);
    timesteps->data[0] = denoiser->sigma_to_t(sigma); // denoiser ???


    TENSOR *noised_output = tensor_copy(input);
    CHECK_TENSOR(noised_output);
    // tensor_zero_(noised_output);  // noised_output = 0

    // timesteps -- c_out = -0.0292, c_in = 0.9996
    // f32 [     1,     1,     1,     1], timesteps
    // f32 [   128,   128,     4,     1], noised_input

    std::vector<TENSOR *> controls;
    // f32 [   128,   128,     4,     1], noised_input
    // f32 [     1,     1,     1,     1], timesteps
    // f32 [  2048,    77,     1,     1], positive_latent
    // f32 [  2816,     1,     1,     1], positive_pooled

    // cond unet->forward ...
    // -----------------------------------------------------------------------------------------------------------------
    // unet->controls = controls;
    // unet->control_strength = control_strength;
    TENSOR *argv1[] = {noised_input, timesteps, positive_latent, positive_pooled, controls_0, controls_1, controls_2, controls_3};
    TENSOR *positive_output = unet->engine_forward(ARRAY_SIZE(argv1), argv1);
    // TENSOR *positive_output = unet_forward(unet, noised_input, timesteps, positive_latent, positive_pooled, controls, control_strength);
    CHECK_TENSOR(positive_output);
    // -----------------------------------------------------------------------------------------------------------------

    // unet->forward ...
    // -----------------------------------------------------------------------------------------------------------------
    TENSOR *argv2[] = {noised_input, timesteps, negative_latent, negative_pooled, controls_0, controls_1, controls_2, controls_3};
    // unet->controls = controls;
    // unet->control_strength = control_strength;
    TENSOR *negative_output = unet->engine_forward(ARRAY_SIZE(argv2), argv2);
    // TENSOR *negative_output = unet_forward(unet, noised_input, timesteps, negative_latent, negative_pooled, controls, control_strength);
    CHECK_TENSOR(negative_output);
    // -----------------------------------------------------------------------------------------------------------------

    // update noised_output 
    {
        float latent_result;
        // config_scale -- 0.0 ==> negative
        // config_scale -- 0.5 ==> (positive + negativee)/2.0
        // config_scale -- 1.0 ==> positive
        // config_scale -- 2.0 ==> 2 * positive  - negative ...
        for (int i = 0; i < n; i++) {
            latent_result = negative_output->data[i] + config_scale * (positive_output->data[i] - negative_output->data[i]);
            noised_output->data[i] = latent_result * c_out + input->data[i] * c_skip; // c_skip == 1.0f
        }
    }

    tensor_destroy(noised_input);
    tensor_destroy(timesteps);
    tensor_destroy(negative_output);
    tensor_destroy(positive_output);

    return noised_output;
}

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>
#define GGML_NN_IMPLEMENTATION
#include <ggml_nn.h>

