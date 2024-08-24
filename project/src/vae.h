/************************************************************************************
***
*** Copyright 2024 Dell(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Sat 24 Aug 2024 03:22:25 PM CST
***
************************************************************************************/

#ifndef __VAE_H__
#define __VAE_H__

#include <ggml_engine.h>
#include <nimage/tensor.h>

/*======= AutoEncoderKL =======*/

struct DownSample {
    int channels;
    int out_channels;

    struct ggml_tensor* op_w; // [out_channels, channels, 3, 3]
    struct ggml_tensor* op_b; // [out_channels,]

    bool vae_downsample = false;

    void create_weight_tensors(struct ggml_context* ctx)
    {
        op_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, out_channels);
        op_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
    }

    void setup_weight_names(const char* prefix)
    {
        if (vae_downsample) {
            ggml_format_name(op_w, "%s%s", prefix, "conv.weight");
            ggml_format_name(op_b, "%s%s", prefix, "conv.bias");
        } else {
            ggml_format_name(op_w, "%s%s", prefix, "op.weight");
            ggml_format_name(op_b, "%s%s", prefix, "op.bias");
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
    {
        // x: [N, channels, h, w]
        struct ggml_tensor* c = NULL;
        if (vae_downsample) {
            c = ggml_pad(ctx, x, 1, 1, 0, 0);
            c = ggml_nn_conv_2d(ctx, c, op_w, op_b, 2, 2, 0, 0, 1, 1);
        } else {
            c = ggml_nn_conv_2d(ctx, x, op_w, op_b, 2, 2, 1, 1, 1, 1);
        }
        return c; // [N, out_channels, h/2, w/2]
    }
};

struct UpSample {
    // hparams
    int channels;
    int out_channels;

    // conv2d params
    struct ggml_tensor* conv_w; // [out_channels, channels, 3, 3]
    struct ggml_tensor* conv_b; // [out_channels,]

    void create_weight_tensors(struct ggml_context* ctx)
    {
        conv_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, channels, out_channels);
        conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(conv_w, "%s%s", prefix, "conv.weight");
        ggml_format_name(conv_b, "%s%s", prefix, "conv.bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
    {
        // x: [N, channels, h, w]
        x = ggml_upscale(ctx, x, 2); // [N, channels, h*2, w*2]
        x = ggml_nn_conv_2d(ctx, x, conv_w, conv_b, 1, 1, 1, 1, 1, 1); // [N, out_channels, h*2, w*2]
        return x;
    }
};

struct ResnetBlock {
    // network hparams
    int in_channels;
    int out_channels;

    // network params
    struct ggml_tensor* norm1_w; // [in_channels, ]
    struct ggml_tensor* norm1_b; // [in_channels, ]

    struct ggml_tensor* conv1_w; // [out_channels, in_channels, 3, 3]
    struct ggml_tensor* conv1_b; // [out_channels, ]

    struct ggml_tensor* norm2_w; // [out_channels, ]
    struct ggml_tensor* norm2_b; // [out_channels, ]

    struct ggml_tensor* conv2_w; // [out_channels, out_channels, 3, 3]
    struct ggml_tensor* conv2_b; // [out_channels, ]

    // nin_shortcut, only if out_channels != in_channels
    struct ggml_tensor* nin_shortcut_w; // [out_channels, in_channels, 1, 1]
    struct ggml_tensor* nin_shortcut_b; // [out_channels, ]

    void create_weight_tensors(struct ggml_context* ctx)
    {
        norm1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        norm1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        conv1_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, in_channels, out_channels);
        conv1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        norm2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        norm2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        conv2_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, out_channels, out_channels);
        conv2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);

        if (out_channels != in_channels) {
            nin_shortcut_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, out_channels);
            nin_shortcut_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_channels);
        }
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(norm1_w, "%s%s", prefix, "norm1.weight");
        ggml_format_name(norm1_b, "%s%s", prefix, "norm1.bias");
        ggml_format_name(conv1_w, "%s%s", prefix, "conv1.weight");
        ggml_format_name(conv1_b, "%s%s", prefix, "conv1.bias");

        ggml_format_name(norm2_w, "%s%s", prefix, "norm2.weight");
        ggml_format_name(norm2_b, "%s%s", prefix, "norm2.bias");
        ggml_format_name(conv2_w, "%s%s", prefix, "conv2.weight");
        ggml_format_name(conv2_b, "%s%s", prefix, "conv2.bias");

        if (out_channels != in_channels) {
            ggml_format_name(nin_shortcut_w, "%s%s", prefix, "nin_shortcut.weight");
            ggml_format_name(nin_shortcut_b, "%s%s", prefix, "nin_shortcut.bias");
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* z)
    {
        // z: [N, in_channels, h, w]

        auto h = ggml_nn_group_norm(ctx, z, norm1_w, norm1_b, 32 /*num_groups*/);
        h = ggml_silu_inplace(ctx, h);
        h = ggml_nn_conv_2d(ctx, h, conv1_w, conv1_b, 1, 1, 1, 1, 1, 1); // [N, out_channels, h, w]
        h = ggml_nn_group_norm(ctx, h, norm2_w, norm2_b, 32 /*num_groups*/);
        h = ggml_silu_inplace(ctx, h);
        // dropout, skip for inference
        h = ggml_nn_conv_2d(ctx, h, conv2_w, conv2_b, 1, 1, 1, 1, 1, 1); // [N, out_channels, h, w]

        // skip connection
        if (out_channels != in_channels) {
            z = ggml_nn_conv_2d(ctx, z, nin_shortcut_w, nin_shortcut_b, 1, 1, 0, 0, 1, 1); // [N, out_channels, h, w]
        }

        h = ggml_add(ctx, h, z);
        return h; // [N, out_channels, h, w]
    }
};

struct AttnBlock {
    int in_channels; // mult * model_channels

    // group norm
    struct ggml_tensor* norm_w; // [in_channels,]
    struct ggml_tensor* norm_b; // [in_channels,]

    // q/k/v
    struct ggml_tensor* q_w; // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* q_b; // [in_channels,]
    struct ggml_tensor* k_w; // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* k_b; // [in_channels,]
    struct ggml_tensor* v_w; // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* v_b; // [in_channels,]

    // proj_out
    struct ggml_tensor* proj_out_w; // [in_channels, in_channels, 1, 1]
    struct ggml_tensor* proj_out_b; // [in_channels,]

    void create_weight_tensors(struct ggml_context* ctx)
    {
        norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        q_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        k_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        v_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

        proj_out_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, in_channels, in_channels);
        proj_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(norm_w, "%s%s", prefix, "norm.weight");
        ggml_format_name(norm_b, "%s%s", prefix, "norm.bias");
        ggml_format_name(q_w, "%s%s", prefix, "q.weight");
        ggml_format_name(q_b, "%s%s", prefix, "q.bias");
        ggml_format_name(k_w, "%s%s", prefix, "k.weight");
        ggml_format_name(k_b, "%s%s", prefix, "k.bias");
        ggml_format_name(v_w, "%s%s", prefix, "v.weight");
        ggml_format_name(v_b, "%s%s", prefix, "v.bias");
        ggml_format_name(proj_out_w, "%s%s", prefix, "proj_out.weight");
        ggml_format_name(proj_out_b, "%s%s", prefix, "proj_out.bias");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
    {
        auto h_ = ggml_nn_group_norm(ctx, x, norm_w, norm_b, 32 /*num_groups*/);

        const int64_t n = h_->ne[3];
        const int64_t c = h_->ne[2];
        const int64_t h = h_->ne[1];
        const int64_t w = h_->ne[0];

        auto q = ggml_nn_conv_2d(ctx, h_, q_w, q_b, 1, 1, 0, 0, 1, 1); // [N, in_channels, h, w]
        auto k = ggml_nn_conv_2d(ctx, h_, k_w, k_b, 1, 1, 0, 0, 1, 1); // [N, in_channels, h, w]
        auto v = ggml_nn_conv_2d(ctx, h_, v_w, v_b, 1, 1, 0, 0, 1, 1); // [N, in_channels, h, w]

        q = ggml_cont(ctx, ggml_permute(ctx, q, 1, 2, 0, 3)); // [N, h, w, in_channels]
        q = ggml_reshape_3d(ctx, q, c, h * w, n); // [N, h * w, in_channels]

        k = ggml_cont(ctx, ggml_permute(ctx, k, 1, 2, 0, 3)); // [N, h, w, in_channels]
        k = ggml_reshape_3d(ctx, k, c, h * w, n); // [N, h * w, in_channels]

        auto w_ = ggml_mul_mat(ctx, k, q); // [N, h * w, h * w]
        w_ = ggml_scale_inplace(ctx, w_, 1.0f / sqrt((float)in_channels));
        w_ = ggml_soft_max_inplace(ctx, w_);

        v = ggml_reshape_3d(ctx, v, h * w, c, n); // [N, in_channels, h * w]
        h_ = ggml_mul_mat(ctx, v, w_); // [N, h * w, in_channels]
        h_ = ggml_cont(ctx, ggml_permute(ctx, h_, 1, 0, 2, 3)); // [N, in_channels, h * w]
        h_ = ggml_reshape_4d(ctx, h_, w, h, c, n); // [N, in_channels, h, w]

        // proj_out
        h_ = ggml_nn_conv_2d(ctx, h_, proj_out_w, proj_out_b, 1, 1, 0, 0, 1, 1); // [N, in_channels, h, w]

        h_ = ggml_add(ctx, h_, x);
        return h_;
    }
};

// ldm.modules.diffusionmodules.model.Encoder
struct Encoder {
    int embed_dim = 4;
    int ch = 128;
    int in_channels = 3;
    int z_channels = 4;
    int ch_mult[4] = { 1, 2, 4, 4 };
    int num_res_blocks = 2;

    struct ggml_tensor* conv_in_w; // [ch, in_channels, 3, 3]
    struct ggml_tensor* conv_in_b; // [ch, ]

    ResnetBlock down_block[4][2];
    DownSample down_sample[3];

    struct
    {
        ResnetBlock block_1;
        AttnBlock attn_1;
        ResnetBlock block_2;
    } mid;

    // block_in = ch * ch_mult[len_mults - 1]
    struct ggml_tensor* norm_out_w; // [block_in, ]
    struct ggml_tensor* norm_out_b; // [block_in, ]

    struct ggml_tensor* conv_out_w; // [embed_dim*2, block_in, 3, 3]
    struct ggml_tensor* conv_out_b; // [embed_dim*2, ]

    Encoder()
    {
        int len_mults = sizeof(ch_mult) / sizeof(int);

        int block_in = 1;
        for (int i = 0; i < len_mults; i++) {
            if (i == 0) {
                block_in = ch;
            } else {
                block_in = ch * ch_mult[i - 1];
            }
            int block_out = ch * ch_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                down_block[i][j].in_channels = block_in;
                down_block[i][j].out_channels = block_out;
                block_in = block_out;
            }
            if (i != len_mults - 1) {
                down_sample[i].channels = block_in;
                down_sample[i].out_channels = block_in;
                down_sample[i].vae_downsample = true;
            }
        }

        mid.block_1.in_channels = block_in;
        mid.block_1.out_channels = block_in;
        mid.attn_1.in_channels = block_in;
        mid.block_2.in_channels = block_in;
        mid.block_2.out_channels = block_in;
    }

    void create_weight_tensors(struct ggml_context* ctx)
    {
        int len_mults = sizeof(ch_mult) / sizeof(int);
        int block_in = ch * ch_mult[len_mults - 1];

        conv_in_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, in_channels, ch);
        conv_in_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ch);

        norm_out_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, block_in);
        norm_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, block_in);

        conv_out_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, block_in, z_channels * 2);
        conv_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, z_channels * 2);

        mid.block_1.create_weight_tensors(ctx);
        mid.attn_1.create_weight_tensors(ctx);
        mid.block_2.create_weight_tensors(ctx);

        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                down_block[i][j].create_weight_tensors(ctx);
            }
            if (i != len_mults - 1) {
                down_sample[i].create_weight_tensors(ctx);
            }
        }
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(norm_out_w, "%s%s", prefix, "norm_out.weight");
        ggml_format_name(norm_out_b, "%s%s", prefix, "norm_out.bias");
        ggml_format_name(conv_in_w, "%s%s", prefix, "conv_in.weight");
        ggml_format_name(conv_in_b, "%s%s", prefix, "conv_in.bias");
        ggml_format_name(conv_out_w, "%s%s", prefix, "conv_out.weight");
        ggml_format_name(conv_out_b, "%s%s", prefix, "conv_out.bias");

        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "mid.block_1.");
        mid.block_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "mid.attn_1.");
        mid.attn_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "mid.block_2.");
        mid.block_2.setup_weight_names(s);

        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                snprintf(s, sizeof(s), "%sdown.%d.block.%d.", prefix, i, j);
                down_block[i][j].setup_weight_names(s);
            }
            if (i != len_mults - 1) {
                snprintf(s, sizeof(s), "%sdown.%d.downsample.", prefix, i);
                down_sample[i].setup_weight_names(s);
            }
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, int argc, struct ggml_tensor* argv[])
    {
        // x[0]: [N, in_channels, h, w]

        // conv_in
        auto x = argv[0];
        auto h = ggml_nn_conv_2d(ctx, x, conv_in_w, conv_in_b, 1, 1, 1, 1, 1, 1); // [N, ch, h, w]
        ggml_set_name(h, "b-start");
        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = 0; i < len_mults; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                h = down_block[i][j].forward(ctx, h);
            }
            if (i != len_mults - 1) {
                h = down_sample[i].forward(ctx, h);
            }
        }

        h = mid.block_1.forward(ctx, h);
        h = mid.attn_1.forward(ctx, h);
        h = mid.block_2.forward(ctx, h); // [N, block_in, h, w]

        h = ggml_nn_group_norm(ctx, h, norm_out_w, norm_out_b, 32 /*num_groups*/);
        h = ggml_silu_inplace(ctx, h);

        // conv_out
        h = ggml_nn_conv_2d(ctx, h, conv_out_w, conv_out_b, 1, 1, 1, 1, 1, 1); // [N, z_channels*2, h, w]

        return h;
    }
};

// ldm.modules.diffusionmodules.model.Decoder
struct Decoder {
    int embed_dim = 4;
    int ch = 128;
    int z_channels = 4;
    int out_ch = 3;
    int num_res_blocks = 2;
    int ch_mult[4] = { 1, 2, 4, 4 };

    struct ggml_tensor* conv_in_w; // [block_in, z_channels, 3, 3]
    struct ggml_tensor* conv_in_b; // [block_in, ]

    struct
    {
        ResnetBlock block_1;
        AttnBlock attn_1;
        ResnetBlock block_2;
    } mid;

    ResnetBlock up_blocks[4][3];
    UpSample up_samples[3];

    struct ggml_tensor* norm_out_w; // [ch *  ch_mult[0], ]
    struct ggml_tensor* norm_out_b; // [ch *  ch_mult[0], ]

    struct ggml_tensor* conv_out_w; // [out_ch, ch *  ch_mult[0], 3, 3]
    struct ggml_tensor* conv_out_b; // [out_ch, ]

    Decoder()
    {
        int len_mults = sizeof(ch_mult) / sizeof(int);
        int block_in = ch * ch_mult[len_mults - 1];

        mid.block_1.in_channels = block_in;
        mid.block_1.out_channels = block_in;
        mid.attn_1.in_channels = block_in;
        mid.block_2.in_channels = block_in;
        mid.block_2.out_channels = block_in;

        for (int i = len_mults - 1; i >= 0; i--) {
            int mult = ch_mult[i];
            int block_out = ch * mult;
            for (int j = 0; j < num_res_blocks + 1; j++) {
                up_blocks[i][j].in_channels = block_in;
                up_blocks[i][j].out_channels = block_out;
                block_in = block_out;
            }
            if (i != 0) {
                up_samples[i - 1].channels = block_in;
                up_samples[i - 1].out_channels = block_in;
            }
        }
    }

    void create_weight_tensors(struct ggml_context* ctx)
    {
        int len_mults = sizeof(ch_mult) / sizeof(int);
        int block_in = ch * ch_mult[len_mults - 1];

        norm_out_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ch * ch_mult[0]);
        norm_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ch * ch_mult[0]);

        conv_in_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, z_channels, block_in);
        conv_in_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, block_in);

        conv_out_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, ch * ch_mult[0], out_ch);
        conv_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_ch);

        mid.block_1.create_weight_tensors(ctx);
        mid.attn_1.create_weight_tensors(ctx);
        mid.block_2.create_weight_tensors(ctx);

        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                up_blocks[i][j].create_weight_tensors(ctx);
            }

            if (i != 0) {
                up_samples[i - 1].create_weight_tensors(ctx);
            }
        }
    }

    void setup_weight_names(const char* prefix)
    {
        char s[GGML_MAX_NAME];

        ggml_format_name(norm_out_w, "%s%s", prefix, "norm_out.weight");
        ggml_format_name(norm_out_b, "%s%s", prefix, "norm_out.bias");
        ggml_format_name(conv_in_w, "%s%s", prefix, "conv_in.weight");
        ggml_format_name(conv_in_b, "%s%s", prefix, "conv_in.bias");
        ggml_format_name(conv_out_w, "%s%s", prefix, "conv_out.weight");
        ggml_format_name(conv_out_b, "%s%s", prefix, "conv_out.bias");

        snprintf(s, sizeof(s), "%s%s", prefix, "mid.block_1.");
        mid.block_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "mid.attn_1.");
        mid.attn_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "mid.block_2.");
        mid.block_2.setup_weight_names(s);

        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                snprintf(s, sizeof(s), "%sup.%d.block.%d.", prefix, i, j);
                up_blocks[i][j].setup_weight_names(s);
            }
            if (i != 0) {
                snprintf(s, sizeof(s), "%sup.%d.upsample.", prefix, i);
                up_samples[i - 1].setup_weight_names(s);
            }
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, int argc, struct ggml_tensor* argv[])
    {
        // z: [N, z_channels, h, w]
        // conv_in
        auto z = argv[0];

        auto h = ggml_nn_conv_2d(ctx, z, conv_in_w, conv_in_b, 1, 1, 1, 1, 1, 1); // [N, block_in, h, w]

        h = mid.block_1.forward(ctx, h);
        h = mid.attn_1.forward(ctx, h);
        h = mid.block_2.forward(ctx, h); // [N, block_in, h, w]

        int len_mults = sizeof(ch_mult) / sizeof(int);
        for (int i = len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                h = up_blocks[i][j].forward(ctx, h);
            }
            if (i != 0) {
                h = up_samples[i - 1].forward(ctx, h);
            }
        }

        // group norm 32
        h = ggml_nn_group_norm(ctx, h, norm_out_w, norm_out_b, 32 /*num_groups*/);
        h = ggml_silu_inplace(ctx, h);

        // conv_out
        h = ggml_nn_conv_2d(ctx, h, conv_out_w, conv_out_b, 1, 1, 1, 1, 1, 1); // [N, out_ch, h, w]
        return h;
    }
};

// ldm.models.autoencoder.AutoencoderKL
struct AutoEncoderKL : GGMLNetwork {
    int embed_dim = 4;
    struct {
        int z_channels = 4;
        int in_channels = 3;
        int out_ch = 3;
        int ch = 128;
        int ch_mult[4] = { 1, 2, 4, 4 };
        int num_res_blocks = 2;
    } dd_config;

    struct ggml_tensor* quant_conv_w; // [2*embed_dim, 2*z_channels, 1, 1]
    struct ggml_tensor* quant_conv_b; // [2*embed_dim, ]

    struct ggml_tensor* post_quant_conv_w; // [z_channels, embed_dim, 1, 1]
    struct ggml_tensor* post_quant_conv_b; // [z_channels, ]

    Encoder encoder;
    Decoder decoder;
    bool encode_flag = true; /// !!!

    size_t get_graph_size()
    {
        return GGML_DEFAULT_GRAPH_SIZE * 10; // 2048 * 10
    }

    AutoEncoderKL()
    {
        assert(sizeof(dd_config.ch_mult) == sizeof(encoder.ch_mult));
        assert(sizeof(dd_config.ch_mult) == sizeof(decoder.ch_mult));

        encoder.embed_dim = embed_dim;
        decoder.embed_dim = embed_dim;
        encoder.ch = dd_config.ch;
        decoder.ch = dd_config.ch;
        encoder.z_channels = dd_config.z_channels;

        decoder.z_channels = dd_config.z_channels;
        encoder.in_channels = dd_config.in_channels;
        decoder.out_ch = dd_config.out_ch;
        encoder.num_res_blocks = dd_config.num_res_blocks;

        int len_mults = sizeof(dd_config.ch_mult) / sizeof(int);
        for (int i = 0; i < len_mults; i++) {
            encoder.ch_mult[i] = dd_config.ch_mult[i];
            decoder.ch_mult[i] = dd_config.ch_mult[i];
        }
    }

    void create_weight_tensors(struct ggml_context* ctx)
    {
        // blocks["quant_conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(embed_dim * factor,
        //                                                              dd_config.z_channels * factor,
        //                                                              {1, 1}));
        quant_conv_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, 2 * dd_config.z_channels, 2 * embed_dim);
        quant_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * embed_dim);
        encoder.create_weight_tensors(ctx);

        post_quant_conv_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, embed_dim, dd_config.z_channels);
        post_quant_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dd_config.z_channels);
        decoder.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char* prefix)
    {
        char s[512];

        ggml_format_name(quant_conv_w, "%s%s", prefix, "quant_conv.weight");
        ggml_format_name(quant_conv_b, "%s%s", prefix, "quant_conv.bias");
        snprintf(s, sizeof(s), "%sencoder.", prefix);
        encoder.setup_weight_names(s);
        // tensors[prefix + "quant_conv.weight"] = quant_conv_w;

        ggml_format_name(post_quant_conv_w, "%s%s", prefix, "post_quant_conv.weight");
        ggml_format_name(post_quant_conv_b, "%s%s", prefix, "post_quant_conv.bias");
        snprintf(s, sizeof(s), "%sdecoder.", prefix);
        decoder.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, int argc, struct ggml_tensor* argv[])
    {
        if (encode_flag) {
            auto h = encoder.forward(ctx, argc, argv); // [N, 2*z_channels, h/8, w/8]
            // quant_conv
            h = ggml_nn_conv_2d(ctx, h, quant_conv_w, quant_conv_b, 1, 1, 0, 0, 1, 1); // [N, 2*embed_dim, h/8, w/8]
            ggml_set_name(h, "b-end");
            return h;
        }

        // decode ...
        auto z = argv[1];
        auto h = ggml_nn_conv_2d(ctx, z, post_quant_conv_w, post_quant_conv_b, 1, 1, 0, 0, 1, 1); // [N, z_channels, h, w]
        ggml_set_name(h, "bench-start");

        argv[0] = h;
        h = decoder.forward(ctx, 1, argv);
        ggml_set_name(h, "bench-end");
        return h;
    }
};


TENSOR *vae_encode(AutoEncoderKL *vae, TENSOR *image);
TENSOR *vae_decode(AutoEncoderKL *vae, TENSOR *latent);

#endif
