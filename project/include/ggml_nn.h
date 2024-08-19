/************************************************************************************
***
*** Copyright 2024 Dell(18588220928g@163.com), All Rights Reserved.
***
*** File Author: Dell, Tue 30 Jan 2024 11:52:34 PM CST
***
************************************************************************************/

#ifndef _GGML_NN_H_
#define _GGML_NN_H_

#include <ggml.h>

#pragma GCC diagnostic ignored "-Wformat-truncation"

// struct ggml_tensor* ggml_nn_identity(struct ggml_context* ctx, struct ggml_tensor* x)
// {
//     return ggml_dup_inplace(ctx, x);
// }



struct ggml_tensor* ggml_nn_conv_2d(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w,
    struct ggml_tensor* b, int s0 = 1, int s1 = 1, int p0 = 0, int p1 = 0, int d0 = 1, int d1 = 1)
{
    x = ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);

    if (b != NULL) {
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
        x = ggml_add(ctx, x, b);
    }

    return x;
}

struct ggml_tensor* ggml_nn_layer_norm(
    struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w, struct ggml_tensor* b)
{
    x = ggml_norm(ctx, x, 1e-6 /*eps*/);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

struct LayerNorm {
    int64_t normalized_shape;

    struct ggml_tensor *w;
    struct ggml_tensor *b;

    void create_weight_tensors(struct ggml_context* ctx, ggml_type wtype=GGML_TYPE_F32) {
        w = ggml_new_tensor_1d(ctx, wtype, normalized_shape);
        b = ggml_new_tensor_1d(ctx, wtype, normalized_shape);
    }

    void setup_weight_names(const char *prefix) {
        ggml_format_name(w, "%s%s", prefix, "weight");
        ggml_format_name(b, "%s%s", prefix, "bias");        
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        return ggml_nn_layer_norm(ctx, x, w, b);
    }
};


struct ggml_tensor* ggml_nn_group_norm(
    struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w, struct ggml_tensor* b, int num_groups = 32)
{
    if (ggml_n_dims(x) >= 3) {
        w = ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1);
        b = ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1);
    }

    x = ggml_group_norm(ctx, x, num_groups);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

// struct ggml_tensor* ggml_nn_group_norm_32(struct ggml_context* ctx, struct ggml_tensor* a)
// {
//     return ggml_group_norm(ctx, a, 32);
// }

struct ggml_tensor* ggml_nn_linear(
    struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w, struct ggml_tensor* b)
{
    x = ggml_mul_mat(ctx, w, x);
    if (b != NULL) {
        x = ggml_add(ctx, x, b);
    }
    return x;
}


struct Linear {
    int64_t in_features;
    int64_t out_features;
    bool bias_flag = true;

    struct ggml_tensor *weight;
    struct ggml_tensor *bias = NULL;

    void create_weight_tensors(struct ggml_context* ctx, ggml_type wtype=GGML_TYPE_F16) {
        weight = ggml_new_tensor_2d(ctx, wtype, in_features, out_features);
        if (bias_flag) {
            bias = ggml_new_tensor_1d(ctx, (wtype == GGML_TYPE_F16)?GGML_TYPE_F32:GGML_TYPE_Q8_0, out_features);
        }
    }

    void setup_weight_names(const char *prefix) {
        ggml_format_name(weight, "%s%s", prefix, "weight");
        if (bias_flag) {
            ggml_format_name(bias, "%s%s", prefix, "bias");        
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        return ggml_nn_linear(ctx, x, weight, bias);
    }
};

struct Conv2d {
    int64_t in_channels;
    int64_t out_channels;
    std::pair<int, int> kernel_size;
    std::pair<int, int> stride = {1, 1};
    std::pair<int, int> padding =  {0, 0};
    std::pair<int, int> dilation = {1, 1};
    bool bias_flag = true;

    struct ggml_tensor *weight;
    struct ggml_tensor *bias = NULL;

    void create_weight_tensors(struct ggml_context* ctx, ggml_type wtype=GGML_TYPE_F16) {
        weight = ggml_new_tensor_4d(ctx, wtype, kernel_size.second, kernel_size.first, in_channels, out_channels);
        if (bias_flag) {
            bias = ggml_new_tensor_1d(ctx, (wtype == GGML_TYPE_F16)?GGML_TYPE_F32:GGML_TYPE_Q8_0, out_channels);
        }        
    }

    void setup_weight_names(const char *prefix) {
        ggml_format_name(weight, "%s%s", prefix, "weight");
        if (bias_flag) {
            ggml_format_name(bias, "%s%s", prefix, "bias");        
        }        
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        return ggml_nn_conv_2d(ctx, x, weight, bias, stride.second, stride.first, padding.second, padding.first, dilation.second, dilation.first);
    }
};

struct GroupNorm32 {
    int64_t num_channels;

    struct ggml_tensor *weight;
    struct ggml_tensor *bias;

    void create_weight_tensors(struct ggml_context* ctx) {
        // norm use GGML_TYPE_F32 !!!
        weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
        bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
    }

    void setup_weight_names(const char *prefix) {
        ggml_format_name(weight, "%s%s", prefix, "weight");
        ggml_format_name(bias, "%s%s", prefix, "bias");        
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        return ggml_nn_group_norm(ctx, x, weight, bias, 32 /*num_groups*/);
    }
};

struct ResBlock {
    int64_t channels;      // model_channels * (1, 1, 1, 2, 2, 4, 4, 4)
    int64_t emb_channels;  // time_embed_dim
    int64_t out_channels;  // mult * model_channels
    std::pair<int, int> kernel_size = {3, 3};
    bool skip_t_emb = false;

    GroupNorm32 in_layers_0;
    Conv2d in_layers_2;
    Linear emb_layer_1;

    GroupNorm32 out_layers_0;
    Conv2d out_layers_3;
    Conv2d skip_connection;

    void create_weight_tensors(struct ggml_context* ctx) {
        std::pair<int, int> padding = {kernel_size.first / 2, kernel_size.second / 2};

        in_layers_0.num_channels = channels; // GroupNorm32
        in_layers_0.create_weight_tensors(ctx);

        // Conv2d
        in_layers_2.in_channels = channels;
        in_layers_2.out_channels = out_channels;
        in_layers_2.kernel_size = kernel_size;
        in_layers_2.padding = padding;
        in_layers_2.create_weight_tensors(ctx);

        if (!skip_t_emb) { // Linear
            emb_layer_1.in_features = emb_channels;
            emb_layer_1.out_features = out_channels;
            emb_layer_1.create_weight_tensors(ctx);
        }

        out_layers_0.num_channels = out_channels; // GroupNorm32
        out_layers_0.create_weight_tensors(ctx);

        // Conv2d
        out_layers_3.in_channels = out_channels;
        out_layers_3.out_channels = out_channels;
        out_layers_3.kernel_size = kernel_size;
        out_layers_3.padding = padding;
        out_layers_3.create_weight_tensors(ctx);

        if (out_channels != channels) { // Conv2d
            skip_connection.in_channels = channels;
            skip_connection.out_channels = out_channels;
            skip_connection.kernel_size = {1, 1};
            skip_connection.padding = {0, 0};
            skip_connection.create_weight_tensors(ctx);
        }
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "in_layers.0.");
        in_layers_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "in_layers.2.");
        in_layers_2.setup_weight_names(s);

        if (!skip_t_emb) {
            snprintf(s, sizeof(s), "%s%s", prefix, "emb_layers.1.");
            emb_layer_1.setup_weight_names(s);
        }

        snprintf(s, sizeof(s), "%s%s", prefix, "out_layers.0.");
        out_layers_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "out_layers.3.");
        out_layers_3.setup_weight_names(s);

        if (out_channels != channels) {
            snprintf(s, sizeof(s), "%s%s", prefix, "skip_connection.");
            skip_connection.setup_weight_names(s);
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* emb = NULL) {
        // [N, c, t, h, w] => [N, c, t, h * w]
        // x: [N, channels, h, w]
        // emb: [N, emb_channels]
        if (emb == NULL) {
            GGML_ASSERT(skip_t_emb);
        }

        // in_layers
        auto h = in_layers_0.forward(ctx, x);
        h = ggml_silu_inplace(ctx, h);
        h = in_layers_2.forward(ctx, h);  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]

        // emb_layers
        if (!skip_t_emb) {
            auto emb_out = ggml_silu(ctx, emb);
            emb_out = emb_layer_1.forward(ctx, emb_out);  // [N, out_channels] if dims == 2 else [N, t, out_channels]
            emb_out = ggml_reshape_4d(ctx, emb_out, 1, 1, emb_out->ne[0], emb_out->ne[1]);  // [N, out_channels, 1, 1]

            h = ggml_add(ctx, h, emb_out);  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]
        }

        // out_layers
        h = out_layers_0.forward(ctx, h);
        h = ggml_silu_inplace(ctx, h);
        // dropout, skip for inference
        h = out_layers_3.forward(ctx, h);

        // skip connection
        if (out_channels != channels) {
            x = skip_connection.forward(ctx, x);  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]
        }

        h = ggml_add(ctx, h, x);
        return h;  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]
    }
};

struct ggml_tensor* ggml_nn_timestep_embedding(
    struct ggml_context* ctx,
    struct ggml_tensor* timesteps,
    int dim,
    int max_period = 10000) {
    return ggml_timestep_embedding(ctx, timesteps, dim, max_period);
}


struct DownSampleBlock {
    int channels;
    int out_channels;

    Conv2d op;

    void create_weight_tensors(struct ggml_context* ctx) {
        op.in_channels = channels;
        op.out_channels = out_channels;
        op.kernel_size = {3, 3};
        op.stride = {2, 2};
        op.padding = {1, 1};
        op.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "op.");
        op.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]
        x = op.forward(ctx, x);
        return x;  // [N, out_channels, h/2, w/2]
    }
};


struct UpSampleBlock {
    int channels;
    int out_channels;

    Conv2d conv;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = channels;
        conv.out_channels = out_channels;
        conv.kernel_size = {3, 3};
        conv.stride = {1, 1};
        conv.padding = {1, 1};
        conv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]
        x = ggml_upscale(ctx, x, 2);  // [N, channels, h*2, w*2]
        x = conv.forward(ctx, x);    // [N, out_channels, h*2, w*2]
        return x;
    }
};


// q: [N * n_head, n_token, d_head]
// k: [N * n_head, n_k, d_head]
// v: [N * n_head, d_head, n_k]
// return: [N * n_head, n_token, d_head]
struct ggml_tensor* ggml_nn_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* q,
    struct ggml_tensor* k,
    struct ggml_tensor* v,
    bool mask = false) {
#if defined(SD_USE_FLASH_ATTENTION) && !defined(SD_USE_CUBLAS) && !defined(SD_USE_METAL)
    struct ggml_tensor* kqv = ggml_flash_attn(ctx, q, k, v, false);  // [N * n_head, n_token, d_head]
#else
    float d_head = (float)q->ne[0];

    struct ggml_tensor* kq = ggml_mul_mat(ctx, k, q);  // [N * n_head, n_token, n_k]
    kq = ggml_scale_inplace(ctx, kq, 1.0f / sqrt(d_head));
    if (mask) {
        kq = ggml_diag_mask_inf_inplace(ctx, kq, 0);
    }
    kq = ggml_soft_max_inplace(ctx, kq);

    struct ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);  // [N * n_head, n_token, d_head]
#endif
    return kqv;
}

#endif // _GGML_NN_H_