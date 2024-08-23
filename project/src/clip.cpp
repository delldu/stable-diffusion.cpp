#include "clip.h"

std::vector<float> timestep_embedding(int height, int width, int dim = 256)
{
    // timesteps -- {h, w}, dim -- 256
    int max_period = 10000;

    if (dim % 2 != 0)
        dim++;

    std::vector<float> embedding(2 * dim, 0.f); // 2 * 256

    int half = dim / 2;
    float log_max = std::log(max_period);
    std::vector<float> freqs(half);
    for (int i = 0; i < half; ++i) {
        freqs[i] = (float)std::exp(-log_max * i / half);
    }

    for (int i = 0; i < half; ++i) {
        float arg = height * freqs[i];
        embedding[i] = std::cos(arg);
        embedding[i + half] = std::sin(arg);
    }

    for (int i = 0; i < half; ++i) {
        float arg = width * freqs[i];
        embedding[dim + i] = std::cos(arg);
        embedding[dim + i + half] = std::sin(arg);
    }

    return embedding; // 256x2 -- 512
}

TENSOR *get_clip_pooled(TENSOR *pooled, int height, int width, int dim=256)
{
    CHECK_TENSOR(pooled);

    int n = pooled->batch * pooled->chan * pooled->height * pooled->width;
    TENSOR *cond_pooled = tensor_create(n + 3*dim, 1, 1, 1);
    CHECK_TENSOR(cond_pooled);

    // Clone pooled ...
    for (int i = 0; i < n; i++) {
        cond_pooled->data[i] = pooled->data[i];
    }

    // Add timestep embeddings ...
    std::vector<float> e = timestep_embedding(height, width, dim);
    for (int i = 0; i < dim; i++) {
        cond_pooled->data[n + i] = e[i];
    }
    e = timestep_embedding(0 /*top*/, 0 /*left*/, dim);
    for (int i = 0; i < dim; i++) {
        cond_pooled->data[n + dim + i] = e[i];
    }
    e = timestep_embedding(height, width, dim);
    for (int i = 0; i < dim; i++) {
        cond_pooled->data[n + 2*dim + i] = e[i];
    }
    e.clear();

    return cond_pooled; // [2816, 1, 1, 1] positive_latent/positive_pooled, negative_latent/negative_pooled
}


std::vector<TENSOR *> clip_encode(TextEncoder *clip, char *text, int height, int width)
{
    #define MAX_LATENTS 16
    #define CHUNK_LEN77 77

    auto tokens_and_weights = clip->tokenize(text, true);
    std::vector<int>& tokens = tokens_and_weights.first; // size() -- 77
    std::vector<float>& weights = tokens_and_weights.second; // size() -- 77

    int64_t start_time  = ggml_time_ms();
    size_t chunk_count = tokens.size() / CHUNK_LEN77;
    if (chunk_count > MAX_LATENTS)
        chunk_count = MAX_LATENTS;

    TENSOR *cond_latent[MAX_LATENTS] = {};
    TENSOR *cond_pooled = NULL;

    // CheckPoint("text = %s, chunk_count = %ld", text.c_str(), chunk_count);
    for (int i = 0; i < chunk_count; i++) {
        std::vector<int> chunk_tokens(tokens.begin() + i * CHUNK_LEN77, tokens.begin() + (i + 1) * CHUNK_LEN77);
        std::vector<float> chunk_weights(weights.begin() + i * CHUNK_LEN77, weights.begin() + (i + 1) * CHUNK_LEN77);

        TENSOR *input_large_14 = tensor_create(CHUNK_LEN77, 1, 1, 1); // f32 [    77,     1,     1,     1]
        if (input_large_14) {
            for (int j = 0; j < CHUNK_LEN77; j++) {
                input_large_14->data[j] = (float)tokens[j];
            }
        } else {
            syslog_error("Allocate memory for input_large_14.");
        }

        // Update chunk_tokens ...
        auto it = std::find(chunk_tokens.begin(), chunk_tokens.end(), EOS_TOKEN_ID);
        if (it != chunk_tokens.end()) {
            std::fill(std::next(it), chunk_tokens.end(), 0);
        }
        size_t max_token_idx = std::min<size_t>(std::distance(chunk_tokens.begin(), it), chunk_tokens.size() - 1);

        TENSOR *input_bigg_14 = tensor_create(CHUNK_LEN77, 1, 1, 1); // f32 [    77,     1,     1,     1]
        if (input_bigg_14) {
            for (int j = 0; j < CHUNK_LEN77; j++) {
                input_bigg_14->data[j] = (float)tokens[j];
            }
        } else {
            syslog_error("Allocate memory for input_bigg_14.");
        }


        clip->max_token_idx = max_token_idx;
        clip->return_pooled = false;
        TENSOR* argv[] = { input_large_14, input_bigg_14 };
        cond_latent[i] = clip->engine_forward(ARRAY_SIZE(argv), argv); // f32 [  2048,    77,     1,     1]

        if (i == 0) {
            // clip->max_token_idx = max_token_idx;
            clip->return_pooled = true;
            // TENSOR* argv[2] = { input_large_14, input_bigg_14 };
            cond_pooled = clip->engine_forward(ARRAY_SIZE(argv), argv); // f32 [  1280,     1,     1,     1]
        }

        tensor_destroy(input_large_14);
        tensor_destroy(input_bigg_14);

        if (cond_latent[i]) {
            for (int b = 0; b < cond_latent[i]->batch; b++) {
                for (int c = 0; c < cond_latent[i]->chan; c++) {
                    // f32 [  2048,    77,     1,     1]
                    // b * 77 + c
                    cond_latent[i]->data[b * cond_latent[i]->chan + c] *= chunk_weights[c];
                }
            }
        } // cond_latent[i]
    }

    TENSOR *condition_latent = tensor_create(cond_latent[0]->batch, cond_latent[0]->chan, chunk_count, 1);
    if (condition_latent) {
        for (int i = 0; i < chunk_count; i++) {
            if (cond_latent[i] == NULL)
                continue;
            int n = cond_latent[i]->batch * cond_latent[i]->chan;
            for (int j = 0; j < n; j++) {
                condition_latent->data[i * n + j] = cond_latent[i]->data[j];
            }
        }
    }
    for (int i = 0; i < chunk_count; i++) {
        tensor_destroy(cond_latent[i]);
    }

    TENSOR *condition_pooled = get_clip_pooled(cond_pooled, height, width, 256 /*dim*/);
    tensor_destroy(cond_pooled);

    return {condition_latent, condition_pooled};

#undef CHUNK_LEN77
#undef MAX_LATENTS
}
