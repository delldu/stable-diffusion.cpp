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
#include <utility>     // std::pair, std::make_pair


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
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

class LayerNorm {
    int64_t normalized_shape;

    struct ggml_tensor *w;
    struct ggml_tensor *b;

    void create_weight_tensors(struct ggml_context* ctx) {
        w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape);
        b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape);
    }

    void setup_weight_names(char *prefix) {
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
    // network hparams
    int64_t in_features;
    int64_t out_features;
    bool bias_flag = true;

    // Weights
    struct tensor *weight;
    struct tensor *bias = NULL;

    void create_weight_tensors(struct ggml_context* ctx) {
        weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, in_features, out_features);
        if (bias_flag) {
            bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_features);
        }
    }

    void setup_weight_names(char *prefix) {
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

    void create_weight_tensors(struct ggml_context* ctx) {
        weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kernel_size.second, kernel_size.first, in_channels, out_channels);
        if (bias_flag) {
            bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }        
    }

    void setup_weight_names(char *prefix) {
        ggml_format_name(weight, "%s%s", prefix, "weight");
        if (bias_flag) {
            ggml_format_name(bias, "%s%s", prefix, "bias");        
        }        
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        return ggml_nn_conv_2d(ctx, x, weight, bias, stride.second, stride.first, padding.second, padding.first, dilation.second, dilation.first);
    }
};


class GroupNorm32 {
    int64_t num_groups = 32;
    int64_t num_channels;

    void create_weight_tensors(struct ggml_context* ctx) {
        // norm use GGML_TYPE_F32 !!!
        weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
        bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
    }

    void setup_weight_names(char *prefix) {
        ggml_format_name(weight, "%s%s", prefix, "weight");
        ggml_format_name(bias, "%s%s", prefix, "bias");        
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        return ggml_nn_group_norm(ctx, x, weight, bias, num_groups);
    }
};

class ResBlock {
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

        // Conv2d
        in_layers_2.in_channels = channels;
        in_layers_2.out_channels = out_channels;
        in_layers_2.kernel_size = kernel_size;
        in_layers_2.padding = padding;

        if (!skip_t_emb) { // Linear
            emb_layer_1.in_features = emb_channels;
            emb_layer_1.out_features = out_channels;
        }

        out_layers_0.num_channels = out_channels; // GroupNorm32

        // Conv2d
        out_layers_3.in_channels = out_channels;
        out_layers_3.out_channels = out_channels;
        out_layers_3.kernel_size = kernel_size;
        out_layers_3.padding = padding;

        if (out_channels != channels) { // Conv2d
            skip_connection.in_channels = channels;
            skip_connection.out_channels = out_channels;
            skip_connection.kernel_size = {1, 1};
            skip_connection.padding = {0, 0};
        }

        in_layers_0.create_weight_tensors(ctx);
        in_layers_2.create_weight_tensors(ctx);
        if (!skip_t_emb) {
            emb_layer_1.create_weight_tensors(ctx);
        }
        out_layers_0.create_weight_tensors(ctx);
        out_layers_3.setup_weight_names(ctx);

        if (out_channels != channels) {
            skip_connection.create_weight_tensors(ctx);
        }
    }

    void setup_weight_names(char *prefix) {
        char s[1024];

        snprintf(s, sizeof(s), "%sin_layers.0", prefix);
        in_layers_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%sin_layers.2", prefix);
        in_layers_2.setup_weight_names(s);

        if (!skip_t_emb) {
            snprintf(s, sizeof(s), "%semb_layers.1", prefix);
            emb_layer_1.setup_weight_names(s);
        }

        snprintf(s, sizeof(s), "%sout_layers.0", prefix);
        out_layers_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%sout_layers.3", prefix);
        out_layers_3.setup_weight_names(s);

        if (out_channels != channels) {
            snprintf(s, sizeof(s), "%sskip_connection", prefix);
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
        auto h = in_layers_0->forward(ctx, x);
        h      = ggml_silu_inplace(ctx, h);
        h      = in_layers_2->forward(ctx, h);  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]

        // emb_layers
        if (!skip_t_emb) {
            auto emb_out = ggml_silu(ctx, emb);
            emb_out = emb_layer_1->forward(ctx, emb_out);  // [N, out_channels] if dims == 2 else [N, t, out_channels]
            emb_out = ggml_reshape_4d(ctx, emb_out, 1, 1, emb_out->ne[0], emb_out->ne[1]);  // [N, out_channels, 1, 1]

            h = ggml_add(ctx, h, emb_out);  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]
        }

        // out_layers
        h = out_layers_0->forward(ctx, h);
        h = ggml_silu_inplace(ctx, h);
        // dropout, skip for inference
        h = out_layers_3->forward(ctx, h);

        // skip connection
        if (out_channels != channels) {
            x = skip_connection->forward(ctx, x);  // [N, out_channels, h, w] if dims == 2 else [N, out_channels, t, h, w]
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



class DownSampleBlock {
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

    void setup_weight_names(char *prefix) {
        char s[1024];
        snprintf(s, sizeof(s), "%s%s", prefix, "op.");
        op.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]
        x = op->forward(ctx, x);
        return x;  // [N, out_channels, h/2, w/2]
    }
};


class UpSampleBlock {
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

    void setup_weight_names(char *prefix) {
        char s[1024];
        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, channels, h, w]
        x = ggml_upscale(ctx, x, 2);  // [N, channels, h*2, w*2]
        x = conv->forward(ctx, x);    // [N, out_channels, h*2, w*2]
        return x;
    }
};



#endif // _GGML_NN_H_