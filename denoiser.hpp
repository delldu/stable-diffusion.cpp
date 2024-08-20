#ifndef __DENOISER_HPP__
#define __DENOISER_HPP__

#include "ggml_extend.hpp"

/*================================================= CompVisDenoiser ==================================================*/
#define TIMESTEPS 1000

void calculate_alphas_cumprod(float* alphas_cumprod)
{
    float start = 0.00085f;
    float end   = 0.0120;

    float ls_sqrt = sqrtf(start);
    float le_sqrt = sqrtf(end);
    float amount  = le_sqrt - ls_sqrt;
    float product = 1.0f;

    for (int i = 0; i < TIMESTEPS; i++) {
        float beta = ls_sqrt + amount * ((float)i / (TIMESTEPS - 1));
        product *= 1.0f - powf(beta, 2.0f);
        alphas_cumprod[i] = product;
    }
}


struct Denoiser {
    // CompVisDenoiser
    float sigma_data = 1.0f;
    float alphas_cumprod[TIMESTEPS]; // ONLY used for sigmas/log_sigmas ...
    float sigmas[TIMESTEPS];
    float log_sigmas[TIMESTEPS];

    void dump() {
        printf("Denoiser\n");
        // printf("alphas_cumprod: ")
        // for (int i = 0; i < TIMESTEPS; i++) {
        //     printf("%.4f%s ", alphas_cumprod[i], (i < TIMESTEPS - 1)? ",":"\n");
        // }
        printf("sigmas: ");
        for (int i = 0; i < TIMESTEPS; i++) {
            printf("%.4f%s ", sigmas[i], (i < TIMESTEPS - 1)? ",":"\n");
        }

        printf("log_sigmas: ");
        for (int i = 0; i < TIMESTEPS; i++) {
            printf("%.4f%s ", log_sigmas[i], (i < TIMESTEPS - 1)? ",":"\n");
        }
    }

    void init() {
        calculate_alphas_cumprod(alphas_cumprod);
        for (int i = 0; i < TIMESTEPS; i++) {
            sigmas[i] = std::sqrt((1 - alphas_cumprod[i]) / alphas_cumprod[i]);
            log_sigmas[i] = std::log(sigmas[i]);
        }
    }

    float sigma_to_t(float sigma) {
        float log_sigma = std::log(sigma);
        std::vector<float> dists;
        dists.reserve(TIMESTEPS);
        for (float v : log_sigmas) {
            dists.push_back(log_sigma - v);
        }

        int low_i = 0;
        {
            for (size_t i = 0; i < TIMESTEPS; i++) {
                if (dists[i] >= 0) {
                    low_i++;
                }
            }
            low_i = std::min(std::max(low_i - 1, 0), TIMESTEPS - 2);
        }
        int high_i = low_i + 1;

        float low  = log_sigmas[low_i];
        float high = log_sigmas[high_i];
        float w = (low - log_sigma) / (low - high);
        w = std::max(0.f, std::min(1.f, w));
        float t = (1.0f - w) * low_i + w * high_i;

        return t;
    }

    float t_to_sigma(float t) {
        int low_i = static_cast<int>(std::floor(t));
        int high_i = static_cast<int>(std::ceil(t));
        float w = t - static_cast<float>(low_i);
        float log_sigma = (1.0f - w) * log_sigmas[low_i] + w * log_sigmas[high_i];
        return std::exp(log_sigma);
    }

    std::vector<float> get_sigmas(uint32_t n) {
        std::vector<float> result;

        int t_max = TIMESTEPS - 1;

        if (n == 0) {
            return result;
        } else if (n == 1) {
            result.push_back(t_to_sigma((float)t_max));
            result.push_back(0);
            return result;
        }

        float step = static_cast<float>(t_max) / static_cast<float>(n - 1);
        for (uint32_t i = 0; i < n; ++i) {
            float t = t_max - step * i;
            result.push_back(t_to_sigma(t));
        }
        result.push_back(0);
        return result;
    }

    std::vector<float> get_scalings(float sigma) {
        float c_out = -sigma;
        float c_in  = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return {c_out, c_in};
    }
};



typedef std::function<ggml_tensor*(ggml_tensor*, float, int)> denoiser_cb_t;

// k diffusion reverse ODE: dx = (x - D(x;\sigma)) / \sigma dt; \sigma(t) = 
// xxxx_1111
static void sample_k_diffusion(denoiser_cb_t model,
                               ggml_context* work_ctx,
                               ggml_tensor* x,
                               std::vector<float> sigmas,
                               std::shared_ptr<RNG> rng) {

    sample_method_t method = EULER_A;

    size_t steps = sigmas.size() - 1;
    // sample_euler_ancestral
    switch (method) {
        case EULER_A: {
            struct ggml_tensor* noise = ggml_dup_tensor(work_ctx, x);
            struct ggml_tensor* d = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                float sigma = sigmas[i];

                // denoise
                ggml_tensor* denoised = model(x, sigma, i + 1); // unet->forward ...

                // d = (x - denoised) / sigma
                {
                    float* vec_d = (float*)d->data;
                    float* vec_x = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;

                    for (int i = 0; i < ggml_nelements(d); i++) {
                        vec_d[i] = (vec_x[i] - vec_denoised[i]) / sigma;
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
                    float* vec_d = (float*)d->data;
                    float* vec_x = (float*)x->data;

                    for (int i = 0; i < ggml_nelements(x); i++) {
                        vec_x[i] = vec_x[i] + vec_d[i] * dt;
                    }
                }

                if (sigmas[i + 1] > 0) { // add noise
                    ggml_tensor_set_f32_randn(noise, rng);
                    {
                        float* vec_x     = (float*)x->data;
                        float* vec_noise = (float*)noise->data;

                        for (int i = 0; i < ggml_nelements(x); i++) {
                            vec_x[i] = vec_x[i] + vec_noise[i] * sigma_up;
                        }
                    }
                }
            }
        } break;
        case EULER:  // Implemented without any sigma churn
        {
            struct ggml_tensor* d = ggml_dup_tensor(work_ctx, x);

            for (int i = 0; i < steps; i++) {
                float sigma = sigmas[i];
                // denoise
                ggml_tensor* denoised = model(x, sigma, i + 1);
                // d = (x - denoised) / sigma
                {
                    float* vec_d = (float*)d->data;
                    float* vec_x = (float*)x->data;
                    float* vec_denoised = (float*)denoised->data;

                    for (int j = 0; j < ggml_nelements(d); j++) {
                        vec_d[j] = (vec_x[j] - vec_denoised[j]) / sigma;
                    }
                }

                float dt = sigmas[i + 1] - sigma;
                // x = x + d * dt
                {
                    float* vec_d = (float*)d->data;
                    float* vec_x = (float*)x->data;

                    for (int j = 0; j < ggml_nelements(x); j++) {
                        vec_x[j] = vec_x[j] + vec_d[j] * dt;
                    }
                }
            }
        } break;

        default:
            LOG_ERROR("Attempting to sample with nonexisting sample method %i", method);
            abort();
    }
}

#endif  // __DENOISER_HPP__
