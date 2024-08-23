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

void set_f32_randn(TENSOR *t, RNG* rng)
{
    int n = t->batch * t->chan * t->height * t->width;
    std::vector<float> random_numbers = rng->randn(n);
    for (int i = 0; i < n; i++) {
        t->data[i] = random_numbers[i];
    }
}


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
                set_f32_randn(noise, rng);
                for (int j = 0; j < n; j++) {
                    x->data[j] = x->data[j] + noise->data[j] * sigma_up; // x = x + sgigma_up * noise
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

