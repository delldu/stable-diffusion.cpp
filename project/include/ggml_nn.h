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

struct ggml_tensor* ggml_nn_identity(struct ggml_context* ctx, struct ggml_tensor* x);
struct ggml_tensor* ggml_nn_conv_2d(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w,
    struct ggml_tensor* b, int s0 /*=1*/, int s1 /*=1*/, int p0 /*=0*/, int p1 /*=0*/, int d0 /*=1*/, int d1 /*=1*/);
struct ggml_tensor* ggml_nn_layer_norm(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w, struct ggml_tensor* b);
struct ggml_tensor* ggml_nn_attention(struct ggml_context* ctx, struct ggml_tensor* q, struct ggml_tensor* k, struct ggml_tensor* v, 
    bool mask);
struct ggml_tensor* ggml_nn_group_norm(
    struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w, struct ggml_tensor* b, int num_groups);
// struct ggml_tensor* ggml_nn_group_norm_32(struct ggml_context* ctx, struct ggml_tensor* a);
struct ggml_tensor* ggml_nn_linear(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w, struct ggml_tensor* b);

// ----------------------------------------------------------------------------------------------------------------------------------------

struct LayerNorm {
    int64_t normalized_shape;

    struct ggml_tensor* w;
    struct ggml_tensor* b;

    void create_weight_tensors(struct ggml_context* ctx)
    {
        w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape);
        b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, normalized_shape);
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(w, "%s%s", prefix, "weight");
        ggml_format_name(b, "%s%s", prefix, "bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
    {
        return ggml_nn_layer_norm(ctx, x, w, b);
    }
};


struct Linear {
    int64_t in_features;
    int64_t out_features;
    bool has_bias = true;

    struct ggml_tensor* weight;
    struct ggml_tensor* bias = NULL;

    void create_weight_tensors(struct ggml_context* ctx, ggml_type wtype = GGML_TYPE_F16)
    {
        weight = ggml_new_tensor_2d(ctx, wtype, in_features, out_features);
        if (has_bias) {
            bias = ggml_new_tensor_1d(ctx, (wtype == GGML_TYPE_Q8_0)? GGML_TYPE_F16 : GGML_TYPE_F32, out_features);
        }
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(weight, "%s%s", prefix, "weight");
        if (has_bias) {
            ggml_format_name(bias, "%s%s", prefix, "bias");
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
    {
        return ggml_nn_linear(ctx, x, weight, bias);
    }
};

struct Conv2d {
    int64_t in_channels;
    int64_t out_channels;
    std::pair<int, int> kernel_size;
    std::pair<int, int> stride = { 1, 1 };
    std::pair<int, int> padding = { 0, 0 };
    std::pair<int, int> dilation = { 1, 1 };
    bool has_bias = true;

    struct ggml_tensor* weight;
    struct ggml_tensor* bias = NULL;

    void create_weight_tensors(struct ggml_context* ctx, ggml_type wtype = GGML_TYPE_F16)
    {
        weight = ggml_new_tensor_4d(ctx, wtype, kernel_size.second, kernel_size.first, in_channels, out_channels);
        if (has_bias) {
            bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(weight, "%s%s", prefix, "weight");
        if (has_bias) {
            ggml_format_name(bias, "%s%s", prefix, "bias");
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
    {
        return ggml_nn_conv_2d(ctx, x, weight, bias, stride.second, stride.first, padding.second, padding.first, dilation.second, dilation.first);
    }
};

struct GroupNorm32 {
    int64_t num_channels;

    struct ggml_tensor* weight;
    struct ggml_tensor* bias;

    void create_weight_tensors(struct ggml_context* ctx, ggml_type wtype = GGML_TYPE_F32)
    {
        // norm use GGML_TYPE_F32 !!!
        weight = ggml_new_tensor_1d(ctx, wtype, num_channels);
        bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_channels);
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(weight, "%s%s", prefix, "weight");
        ggml_format_name(bias, "%s%s", prefix, "bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
    {
        return ggml_nn_group_norm(ctx, x, weight, bias, 32 /*num_groups*/);
    }
};
#endif // _GGML_NN_H_

#ifdef GGML_NN_IMPLEMENTATION
struct ggml_tensor* ggml_nn_identity(struct ggml_context* ctx, struct ggml_tensor* x)
{
    return ggml_dup_inplace(ctx, x);
}

struct ggml_tensor* ggml_nn_conv_2d(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w,
    struct ggml_tensor* b, int s0 /*=1*/, int s1 /*=1*/, int p0 /*=0*/, int p1 /*=0*/, int d0 /*=1*/, int d1 /*=1*/)
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

// q: [N * n_head, n_token, d_head]
// k: [N * n_head, n_k, d_head]
// v: [N * n_head, d_head, n_k]
// return: [N * n_head, n_token, d_head]
struct ggml_tensor* ggml_nn_attention(
    struct ggml_context* ctx,
    struct ggml_tensor* q,
    struct ggml_tensor* k,
    struct ggml_tensor* v,
    bool mask /* = false*/)
{
#if defined(SD_USE_FLASH_ATTENTION) && !defined(SD_USE_CUBLAS) && !defined(SD_USE_METAL)
    struct ggml_tensor* kqv = ggml_flash_attn(ctx, q, k, v, false); // [N * n_head, n_token, d_head]
#else
    float d_head = (float)q->ne[0];

    struct ggml_tensor* kq = ggml_mul_mat(ctx, k, q); // [N * n_head, n_token, n_k]
    kq = ggml_scale_inplace(ctx, kq, 1.0f / sqrt(d_head));
    if (mask) {
        kq = ggml_diag_mask_inf_inplace(ctx, kq, 0);
    }
    kq = ggml_soft_max_inplace(ctx, kq);

    struct ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq); // [N * n_head, n_token, d_head]
#endif
    return kqv;
}

struct ggml_tensor* ggml_nn_group_norm(
    struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w, struct ggml_tensor* b, int num_groups)
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


#endif // GGML_NN_IMPLEMENTATION