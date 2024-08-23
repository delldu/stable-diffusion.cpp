#ifndef __DENOISER_H__
#define __DENOISER_H__

#include <functional>
#include <random>
#include <vector>

#include "ggml_engine.h"

enum sample_method_t {
    EULER_A,
    EULER,
    N_SAMPLE_METHODS
};

/*================================================= CompVisDenoiser ==================================================*/
#define TIMESTEPS 1000

void calculate_alphas_cumprod(float* alphas_cumprod);


class RNG {
private:
    std::default_random_engine generator;

public:
    void manual_seed(uint64_t seed)
    {
        generator.seed((unsigned int)seed);
    }

    std::vector<float> randn(uint32_t n)
    {
        std::vector<float> result;
        float mean = 0.0;
        float stddev = 1.0;
        std::normal_distribution<float> distribution(mean, stddev);
        for (uint32_t i = 0; i < n; i++) {
            float random_number = distribution(generator);
            result.push_back(random_number);
        }
        return result;
    }
};

struct Denoiser {
    // CompVisDenoiser
    float sigma_data = 1.0f;
    float alphas_cumprod[TIMESTEPS];    // ONLY used for sigmas/log_sigmas ...
    float sigmas[TIMESTEPS];            // 0.0292, ... , 14.6147
    float log_sigmas[TIMESTEPS];        // -3.5347, ..., 2.6820

    void dump()
    {
        printf("Denoiser\n");
        printf("sigmas: ");
        for (int i = 0; i < TIMESTEPS; i++) {
            printf("%.4f%s ", sigmas[i], (i < TIMESTEPS - 1) ? "," : "\n");
        }

        printf("log_sigmas: ");
        for (int i = 0; i < TIMESTEPS; i++) {
            printf("%.4f%s ", log_sigmas[i], (i < TIMESTEPS - 1) ? "," : "\n");
        }
    }

    void init()
    {
        calculate_alphas_cumprod(alphas_cumprod);
        for (int i = 0; i < TIMESTEPS; i++) {
            sigmas[i] = std::sqrt((1 - alphas_cumprod[i]) / alphas_cumprod[i]);
            log_sigmas[i] = std::log(sigmas[i]);
        }
    }

    float sigma_to_t(float sigma)
    {
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

        float low = log_sigmas[low_i];
        float high = log_sigmas[high_i];
        float w = (low - log_sigma) / (low - high);
        w = std::max(0.f, std::min(1.f, w));
        float t = (1.0f - w) * low_i + w * high_i;

        return t;
    }

    float t_to_sigma(float t)
    {
        int low_i = static_cast<int>(std::floor(t));
        int high_i = static_cast<int>(std::ceil(t));
        float w = t - static_cast<float>(low_i);
        float log_sigma = (1.0f - w) * log_sigmas[low_i] + w * log_sigmas[high_i];
        return std::exp(log_sigma);
    }

    std::vector<float> get_sigmas(uint32_t n)
    {
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

    std::vector<float> get_scalings(float sigma)
    {
        float c_out = -sigma;
        float c_in = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return { c_out, c_in };
    }
};


typedef std::function<TENSOR *(TENSOR*, float, int)> denoiser_cb_t;
void k_sample(denoiser_cb_t denoise_model, TENSOR *x, std::vector<float> sigmas, RNG* rng);
void set_f32_randn(TENSOR *t, RNG* rng);


#endif // __DENOISER_H__
