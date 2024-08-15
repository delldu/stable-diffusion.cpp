#ifndef __UNET_HPP__
#define __UNET_HPP__

#define GGML_ENGINE_IMPLEMENTATION
#include "ggml_engine.h"
#include "ggml_nn.h"

#include <utility>     // std::pair, std::make_pair

/*==================================================== UnetModel =====================================================*/

struct GEGLU {
    int64_t dim_in;
    int64_t dim_out;

    // Linear proj;

    struct ggml_tensor* w;
    struct ggml_tensor* b;


    void create_weight_tensors(struct ggml_context* ctx) {
        w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, dim_in, dim_out * 2);
        b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim_out * 2);
    }

    void setup_weight_names(const char *prefix) {
        ggml_format_name(w, "%s%s", prefix, "weight");
        ggml_format_name(b, "%s%s", prefix, "bias");        
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [ne3, ne2, ne1, dim_in]
        // return: [ne3, ne2, ne1, dim_out]
        // struct ggml_tensor* w = proj.weight;
        // struct ggml_tensor* b = proj.bias;

        auto x_w    = ggml_view_2d(ctx, w, w->ne[0], w->ne[1] / 2, w->nb[1], 0);                        // [dim_out, dim_in]
        auto x_b    = ggml_view_1d(ctx, b, b->ne[0] / 2, 0);                                            // [dim_out, dim_in]
        auto gate_w = ggml_view_2d(ctx, w, w->ne[0], w->ne[1] / 2, w->nb[1], w->nb[1] * w->ne[1] / 2);  // [dim_out, ]
        auto gate_b = ggml_view_1d(ctx, b, b->ne[0] / 2, b->nb[0] * b->ne[0] / 2);                      // [dim_out, ]

        auto x_in = x;
        x         = ggml_nn_linear(ctx, x_in, x_w, x_b);        // [ne3, ne2, ne1, dim_out]
        auto gate = ggml_nn_linear(ctx, x_in, gate_w, gate_b);  // [ne3, ne2, ne1, dim_out]

        gate = ggml_gelu_inplace(ctx, gate);

        x = ggml_mul(ctx, x, gate);  // [ne3, ne2, ne1, dim_out]

        return x;
    }
};

struct FeedForward {
    int64_t dim;
    int64_t dim_out;

    GEGLU net_0;
    Linear net_2;

    void create_weight_tensors(struct ggml_context* ctx) {
        int64_t mult = 4;
        int64_t inner_dim = dim * mult;

        net_0.dim_in = dim;
        net_0.dim_out = inner_dim;
        net_0.create_weight_tensors(ctx);

        net_2.in_features = inner_dim;
        net_2.out_features = dim_out;
        net_2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[1024];
        snprintf(s, sizeof(s), "%s%s", prefix, "net.0.");
        net_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "net.2.");
        net_2.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [ne3, ne2, ne1, dim]
        // return: [ne3, ne2, ne1, dim_out]
        x = net_0.forward(ctx, x);  // [ne3, ne2, ne1, inner_dim]
        x = net_2.forward(ctx, x);  // [ne3, ne2, ne1, dim_out]
        return x;
    }
};

struct CrossAttention {
    int64_t query_dim;
    int64_t context_dim;
    int64_t n_head = 20;
    int64_t d_head = 64;

    Linear to_q;
    Linear to_k;
    Linear to_v;
    Linear to_out_0;

    void create_weight_tensors(struct ggml_context* ctx) {
        int64_t inner_dim = d_head * n_head;

        to_q.in_features = query_dim;
        to_q.out_features = inner_dim;
        to_q.bias_flag = false;
        to_q.create_weight_tensors(ctx);

        to_k.in_features = context_dim;
        to_k.out_features = inner_dim;
        to_k.bias_flag = false;
        to_k.create_weight_tensors(ctx);

        to_v.in_features = context_dim;
        to_v.out_features = inner_dim;
        to_v.bias_flag = false;
        to_v.create_weight_tensors(ctx);

        to_out_0.in_features = inner_dim;
        to_out_0.out_features = query_dim;
        to_out_0.bias_flag = false;
        to_out_0.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[1024];
        snprintf(s, sizeof(s), "%s%s", prefix, "to_q.");
        to_q.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "to_k.");
        to_k.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "to_v.");
        to_v.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "to_out.0.");
        to_out_0.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* context) {
        // x: [N, n_token, query_dim]
        // context: [N, n_context, context_dim]
        // return: [N, n_token, query_dim]
        int64_t n         = x->ne[2];
        int64_t n_token   = x->ne[1];
        int64_t n_context = context->ne[1];
        int64_t inner_dim = d_head * n_head;

        auto q = to_q.forward(ctx, x);                                 // [N, n_token, inner_dim]
        q      = ggml_reshape_4d(ctx, q, d_head, n_head, n_token, n);   // [N, n_token, n_head, d_head]
        q      = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));      // [N, n_head, n_token, d_head]
        q      = ggml_reshape_3d(ctx, q, d_head, n_token, n_head * n);  // [N * n_head, n_token, d_head]

        auto k = to_k.forward(ctx, context);                             // [N, n_context, inner_dim]
        k      = ggml_reshape_4d(ctx, k, d_head, n_head, n_context, n);   // [N, n_context, n_head, d_head]
        k      = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));        // [N, n_head, n_context, d_head]
        k      = ggml_reshape_3d(ctx, k, d_head, n_context, n_head * n);  // [N * n_head, n_context, d_head]

        auto v = to_v.forward(ctx, context);                             // [N, n_context, inner_dim]
        v      = ggml_reshape_4d(ctx, v, d_head, n_head, n_context, n);   // [N, n_context, n_head, d_head]
        v      = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));        // [N, n_head, d_head, n_context]
        v      = ggml_reshape_3d(ctx, v, n_context, d_head, n_head * n);  // [N * n_head, d_head, n_context]

        auto kqv = ggml_nn_attention(ctx, q, k, v, false);  // [N * n_head, n_token, d_head]
        kqv      = ggml_reshape_4d(ctx, kqv, d_head, n_token, n_head, n);
        kqv      = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // [N, n_token, n_head, d_head]

        x = ggml_reshape_3d(ctx, kqv, d_head * n_head, n_token, n);  // [N, n_token, inner_dim]

        x = to_out_0.forward(ctx, x);  // [N, n_token, query_dim]
        return x;
    }
};


struct BasicTransformerBlock {
    int64_t dim;
    int64_t n_head = 20;
    int64_t d_head = 64;
    int64_t context_dim;
    bool ff_in_flag = false;

    CrossAttention attn1;
    FeedForward ff;
    CrossAttention attn2;

    LayerNorm norm1;
    LayerNorm norm2;
    LayerNorm norm3;

    // LayerNorm norm_in;
    // FeedForward ff_in;

    void create_weight_tensors(struct ggml_context* ctx) {
        attn1.query_dim = dim;
        attn1.context_dim = dim;
        attn1.n_head = n_head;
        attn1.d_head = d_head;
        attn1.create_weight_tensors(ctx);

        ff.dim = dim;
        ff.dim_out = dim;
        ff.create_weight_tensors(ctx);

        attn2.query_dim = dim;
        attn2.context_dim = context_dim;
        attn2.n_head = n_head;
        attn2.d_head = d_head;
        attn2.create_weight_tensors(ctx);

        norm1.normalized_shape = dim;
        norm1.create_weight_tensors(ctx);

        norm2.normalized_shape = dim;
        norm2.create_weight_tensors(ctx);

        norm3.normalized_shape = dim;
        norm3.create_weight_tensors(ctx);

        // if (ff_in_flag) {
        //     norm_in.normalized_shape = dim;
        //     norm_in.create_weight_tensors(ctx);

        //     ff_in.dim = dim;
        //     ff_in.dim_out = dim;
        //     ff_in.create_weight_tensors(ctx);
        // }
    }

    void setup_weight_names(const char *prefix) {
        char s[1024];

        snprintf(s, sizeof(s), "%s%s", prefix, "attn1.");
        attn1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "ff.");
        ff.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "attn2.");
        attn2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "norm1.");
        norm1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "norm2.");
        norm2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "norm3.");
        norm3.setup_weight_names(s);

        // if (ff_in_flag) {
        //     snprintf(s, sizeof(s), "%s%s", prefix, "norm_in.");
        //     norm_in.setup_weight_names(s);

        //     snprintf(s, sizeof(s), "%s%s", prefix, "ff_in.");
        //     ff_in.setup_weight_names(s);
        // }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* context) {
        // x: [N, n_token, query_dim]
        // context: [N, n_context, context_dim]
        // return: [N, n_token, query_dim]

        // if (ff_in_flag) {
        //     auto x_skip = x;
        //     x           = norm_in.forward(ctx, x);
        //     x           = ff_in.forward(ctx, x);
        //     // self.is_res is always True
        //     x = ggml_add(ctx, x, x_skip);
        // }

        auto r = x;
        x      = norm1.forward(ctx, x);
        x      = attn1.forward(ctx, x, x);  // self-attention
        x      = ggml_add(ctx, x, r);
        r      = x;
        x      = norm2.forward(ctx, x);
        x      = attn2.forward(ctx, x, context);  // cross-attention
        x      = ggml_add(ctx, x, r);
        r      = x;
        x      = norm3.forward(ctx, x);
        x      = ff.forward(ctx, x);
        x      = ggml_add(ctx, x, r);

        return x;
    }
};

struct SpatialTransformer {
    int64_t in_channels;  // mult * model_channels
    int64_t n_head;
    int64_t d_head;
    int64_t depth       = 10; // max depth is 10 ?
    int64_t context_dim = 2048;  // hidden_size, 1024 for VERSION_2_x

    GroupNorm32 norm;
    Conv2d proj_in;
    BasicTransformerBlock transformer_blocks[10];
    Conv2d proj_out;

    void create_weight_tensors(struct ggml_context* ctx) {
        int64_t inner_dim = n_head * d_head;  // in_channels

        norm.num_channels = in_channels;
        norm.create_weight_tensors(ctx);

        proj_in.in_channels = in_channels;
        proj_in.out_channels = inner_dim;
        proj_in.kernel_size = {1, 1};
        proj_in.create_weight_tensors(ctx);

        for (int i = 0; i < depth; i++) {
            transformer_blocks[i].dim = inner_dim;
            transformer_blocks[i].n_head = n_head;
            transformer_blocks[i].d_head = d_head;
            transformer_blocks[i].context_dim = context_dim;
            transformer_blocks[i].create_weight_tensors(ctx);
        }

        proj_out.in_channels = inner_dim;
        proj_out.out_channels = in_channels;
        proj_out.kernel_size = {1, 1};
        proj_out.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[1024];

        snprintf(s, sizeof(s), "%s%s", prefix, "norm.");
        norm.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "proj_in.");
        proj_in.setup_weight_names(s);

        for (int i = 0; i < depth; i++) {
            snprintf(s, sizeof(s), "%s%s%d.", prefix, "transformer_blocks.", i);
            transformer_blocks[i].setup_weight_names(s);
        }

        snprintf(s, sizeof(s), "%s%s", prefix, "proj_out.");
        proj_out.setup_weight_names(s);
    }


    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* context) {
        // x: [N, in_channels, h, w]
        // context: [N, max_position(aka n_token), hidden_size(aka context_dim)]
        // auto norm     = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm"]);
        // auto proj_in  = std::dynamic_pointer_cast<Conv2d>(blocks["proj_in"]);
        // auto proj_out = std::dynamic_pointer_cast<Conv2d>(blocks["proj_out"]);

        auto x_in         = x;
        int64_t n         = x->ne[3];
        int64_t h         = x->ne[1];
        int64_t w         = x->ne[0];
        int64_t inner_dim = n_head * d_head;

        x = norm.forward(ctx, x);
        x = proj_in.forward(ctx, x);  // [N, inner_dim, h, w]

        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));  // [N, h, w, inner_dim]
        x = ggml_reshape_3d(ctx, x, inner_dim, w * h, n);      // [N, h * w, inner_dim]

        for (int i = 0; i < depth; i++) {
            // std::string name       = "transformer_blocks." + std::to_string(i);
            auto transformer_block = transformer_blocks[i]; // std::dynamic_pointer_cast<BasicTransformerBlock>(blocks[name]);
            x = transformer_block.forward(ctx, x, context);
        }

        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));  // [N, inner_dim, h * w]
        x = ggml_reshape_4d(ctx, x, w, h, inner_dim, n);       // [N, inner_dim, h, w]

        // proj_out
        x = proj_out.forward(ctx, x);  // [N, in_channels, h, w]

        x = ggml_add(ctx, x, x_in);
        return x;
    }
};


// ldm.modules.diffusionmodules.openaimodel.UNetModel
struct UNetModel : GGMLNetwork {
    int in_channels                        = 4;
    int out_channels                       = 4;
    int num_res_blocks                     = 2;
    std::vector<int> channel_mult          = {1, 2, 4};
    int time_embed_dim                     = 1280;  // model_channels*4
    int num_head_channels                  = 64;   // channels
    int context_dim                        = 2048;  // 1024 for VERSION_2_x, 2048 for VERSION_XL
    int model_channels  = 320;
    int adm_in_channels = 2816;  // only for VERSION_XL/SVD

    // ---------------------------------------------------------
    std::vector<struct ggml_tensor*> controls = {};
    float control_strength  = 0.f;

    // ---------------------------------------------------------
    Linear time_embed_0;
    Linear time_embed_2;

    Linear label_emb_0_0;
    Linear label_emb_0_2;

    // input blocks;
    Conv2d input_blocks_0_0;
    ResBlock input_blocks_1_0;
    ResBlock input_blocks_2_0;
    ResBlock input_blocks_4_0;
    ResBlock input_blocks_5_0;
    ResBlock input_blocks_7_0;
    ResBlock input_blocks_8_0;
    SpatialTransformer input_blocks_4_1;
    SpatialTransformer input_blocks_5_1;
    SpatialTransformer input_blocks_7_1;
    SpatialTransformer input_blocks_8_1;
    DownSampleBlock input_blocks_3_0;
    DownSampleBlock input_blocks_6_0;

    // middle blocks 
    ResBlock middle_block_0;
    SpatialTransformer middle_block_1;
    ResBlock middle_block_2;

    // output blocks
    ResBlock output_blocks_0_0;
    ResBlock output_blocks_1_0;
    ResBlock output_blocks_2_0;
    ResBlock output_blocks_3_0;
    ResBlock output_blocks_4_0;
    ResBlock output_blocks_5_0;
    ResBlock output_blocks_6_0;
    ResBlock output_blocks_7_0;
    ResBlock output_blocks_8_0;

    SpatialTransformer output_blocks_0_1;
    SpatialTransformer output_blocks_1_1;
    SpatialTransformer output_blocks_2_1;
    SpatialTransformer output_blocks_3_1;
    SpatialTransformer output_blocks_4_1;
    SpatialTransformer output_blocks_5_1;
    UpSampleBlock output_blocks_2_2;
    UpSampleBlock output_blocks_5_2;

    GroupNorm32 out_0;
    Conv2d out_2;


    size_t get_graph_size()
    {
        return GGML_DEFAULT_GRAPH_SIZE * 10; // 2048 * 10
    }

    void create_weight_tensors(struct ggml_context* ctx) {
        // Conv2d
        input_blocks_0_0.in_channels = in_channels;
        input_blocks_0_0.out_channels = model_channels;
        input_blocks_0_0.kernel_size = {3, 3};
        input_blocks_0_0.padding = {1, 1};
        input_blocks_0_0.create_weight_tensors(ctx);

        // ResBlock input_blocks_1_0;
        // name = input_blocks.1.0, ch = 320, time_embed_dim = 1280, out_channels = 320
        // name = input_blocks.2.0, ch = 320, time_embed_dim = 1280, out_channels = 320
        // name = input_blocks.4.0, ch = 320, time_embed_dim = 1280, out_channels = 640
        // name = input_blocks.5.0, ch = 640, time_embed_dim = 1280, out_channels = 640
        // name = input_blocks.7.0, ch = 640, time_embed_dim = 1280, out_channels = 1280
        // name = input_blocks.8.0, ch = 1280, time_embed_dim = 1280, out_channels = 1280
        input_blocks_1_0.channels = model_channels;
        input_blocks_1_0.emb_channels = time_embed_dim;
        input_blocks_1_0.out_channels = model_channels;
        input_blocks_1_0.create_weight_tensors(ctx);

        input_blocks_2_0.channels = model_channels;
        input_blocks_2_0.emb_channels = time_embed_dim;
        input_blocks_2_0.out_channels = model_channels;
        input_blocks_2_0.create_weight_tensors(ctx);

        input_blocks_4_0.channels = model_channels;
        input_blocks_4_0.emb_channels = time_embed_dim;
        input_blocks_4_0.out_channels = 2 * model_channels;
        input_blocks_4_0.create_weight_tensors(ctx);

        input_blocks_5_0.channels = 2 * model_channels;
        input_blocks_5_0.emb_channels = time_embed_dim;
        input_blocks_5_0.out_channels = 2 * model_channels;
        input_blocks_5_0.create_weight_tensors(ctx);

        input_blocks_7_0.channels = 2 * model_channels;
        input_blocks_7_0.emb_channels = time_embed_dim;
        input_blocks_7_0.out_channels = 4 * model_channels;
        input_blocks_7_0.create_weight_tensors(ctx);

        input_blocks_8_0.channels = 4 * model_channels;
        input_blocks_8_0.emb_channels = time_embed_dim;
        input_blocks_8_0.out_channels = 4 * model_channels;
        input_blocks_8_0.create_weight_tensors(ctx);

        // SpatialTransformer
        // channel_mult -- {1, 2, 4}, transformer_depth -- {1, 2, 10}
        // input_blocks.4.1, ch = 640, n_head = 10, d_head = 64, transformer_depth[i]=2, context_dim = 2048
        // input_blocks.5.1, ch = 640, n_head = 10, d_head = 64, transformer_depth[i]=2, context_dim = 2048
        // input_blocks.7.1, ch = 1280, n_head = 20, d_head = 64, transformer_depth[i]=10, context_dim = 2048
        // input_blocks.8.1, ch = 1280, n_head = 20, d_head = 64, transformer_depth[i]=10, context_dim = 2048
        input_blocks_4_1.in_channels = 2 * model_channels;
        input_blocks_4_1.n_head = 10;
        input_blocks_4_1.d_head = num_head_channels; // 64
        input_blocks_4_1.depth = 2;
        input_blocks_4_1.context_dim = context_dim; // 2048
        input_blocks_4_1.create_weight_tensors(ctx);

        input_blocks_5_1.in_channels = 2 * model_channels;
        input_blocks_5_1.n_head = 10;
        input_blocks_5_1.d_head = num_head_channels; // 64
        input_blocks_5_1.depth = 2;
        input_blocks_5_1.context_dim = context_dim; // 2048
        input_blocks_5_1.create_weight_tensors(ctx);

        input_blocks_7_1.in_channels = 4 * model_channels;
        input_blocks_7_1.n_head = 20;
        input_blocks_7_1.d_head = num_head_channels; // 64
        input_blocks_7_1.depth = 10;
        input_blocks_7_1.context_dim = context_dim; // 2048
        input_blocks_7_1.create_weight_tensors(ctx);

        input_blocks_8_1.in_channels = 4 * model_channels;
        input_blocks_8_1.n_head = 20;
        input_blocks_8_1.d_head = num_head_channels; // 64
        input_blocks_8_1.depth = 10;
        input_blocks_8_1.context_dim = context_dim; // 2048
        input_blocks_8_1.create_weight_tensors(ctx);

        // DownSampleBlock
        // name = input_blocks.3.0, ch = 320
        // name = input_blocks.6.0, ch = 640
        input_blocks_3_0.channels = model_channels;
        input_blocks_3_0.out_channels = model_channels;
        input_blocks_3_0.create_weight_tensors(ctx);

        input_blocks_6_0.channels = 2 * model_channels;
        input_blocks_6_0.out_channels = 2 * model_channels;
        input_blocks_6_0.create_weight_tensors(ctx);


        // ResBlock
        middle_block_0.channels = 4 * model_channels;
        middle_block_0.emb_channels = time_embed_dim;
        middle_block_0.out_channels = 4 * model_channels;
        middle_block_0.create_weight_tensors(ctx);

        // SpatialTransformer
        middle_block_1.in_channels = 2 * model_channels;
        middle_block_1.n_head = 20;
        middle_block_1.d_head = num_head_channels; // 64
        middle_block_1.depth = 10;
        middle_block_1.context_dim = context_dim; // 2048
        middle_block_1.create_weight_tensors(ctx);

        // ResBlock
        middle_block_2.channels = 4 * model_channels;
        middle_block_2.emb_channels = time_embed_dim;
        middle_block_2.out_channels = 4 * model_channels;
        middle_block_2.create_weight_tensors(ctx);

        // name = output_blocks.0.0, channels = 2560, time_embed_dim = 1280, out_channels=1280
        // name = output_blocks.1.0, channels = 2560, time_embed_dim = 1280, out_channels=1280
        // name = output_blocks.2.0, channels = 1920, time_embed_dim = 1280, out_channels=1280
        // name = output_blocks.3.0, channels = 1920, time_embed_dim = 1280, out_channels=640
        // name = output_blocks.4.0, channels = 1280, time_embed_dim = 1280, out_channels=640
        // name = output_blocks.5.0, channels = 960, time_embed_dim = 1280, out_channels=640
        // name = output_blocks.6.0, channels = 960, time_embed_dim = 1280, out_channels=320
        // name = output_blocks.7.0, channels = 640, time_embed_dim = 1280, out_channels=320
        // name = output_blocks.8.0, channels = 640, time_embed_dim = 1280, out_channels=320
        output_blocks_0_0.channels = 8 * model_channels;
        output_blocks_0_0.emb_channels = time_embed_dim;
        output_blocks_0_0.out_channels = 4 * model_channels;
        output_blocks_0_0.create_weight_tensors(ctx);

        output_blocks_1_0.channels = 8 * model_channels;
        output_blocks_1_0.emb_channels = time_embed_dim;
        output_blocks_1_0.out_channels = 4 * model_channels;
        output_blocks_1_0.create_weight_tensors(ctx);

        output_blocks_2_0.channels = 6 * model_channels;
        output_blocks_2_0.emb_channels = time_embed_dim;
        output_blocks_2_0.out_channels = 4 * model_channels;
        output_blocks_2_0.create_weight_tensors(ctx);

        output_blocks_3_0.channels = 6 * model_channels;
        output_blocks_3_0.emb_channels = time_embed_dim;
        output_blocks_3_0.out_channels = 2 * model_channels;
        output_blocks_3_0.create_weight_tensors(ctx);

        output_blocks_4_0.channels = 4 * model_channels;
        output_blocks_4_0.emb_channels = time_embed_dim;
        output_blocks_4_0.out_channels = 2 * model_channels;
        output_blocks_4_0.create_weight_tensors(ctx);

        output_blocks_5_0.channels = 3 * model_channels;
        output_blocks_5_0.emb_channels = time_embed_dim;
        output_blocks_5_0.out_channels = 2 * model_channels;
        output_blocks_5_0.create_weight_tensors(ctx);

        output_blocks_6_0.channels = 3 * model_channels;
        output_blocks_6_0.emb_channels = time_embed_dim;
        output_blocks_6_0.out_channels = 1 * model_channels;
        output_blocks_6_0.create_weight_tensors(ctx);

        output_blocks_7_0.channels = 2 * model_channels;
        output_blocks_7_0.emb_channels = time_embed_dim;
        output_blocks_7_0.out_channels = 1 * model_channels;
        output_blocks_7_0.create_weight_tensors(ctx);

        output_blocks_8_0.channels = 2 * model_channels;
        output_blocks_8_0.emb_channels = time_embed_dim;
        output_blocks_8_0.out_channels = 1 * model_channels;
        output_blocks_8_0.create_weight_tensors(ctx);


        // SpatialTransformer
        // name = output_blocks.0.1, ch = 1280, n_head = 20, d_head = 64, depth=10, context_dim=2048
        // name = output_blocks.1.1, ch = 1280, n_head = 20, d_head = 64, depth=10, context_dim=2048
        // name = output_blocks.2.1, ch = 1280, n_head = 20, d_head = 64, depth=10, context_dim=2048
        // name = output_blocks.3.1, ch = 640, n_head = 10, d_head = 64, depth=2, context_dim=2048
        // name = output_blocks.4.1, ch = 640, n_head = 10, d_head = 64, depth=2, context_dim=2048
        // name = output_blocks.5.1, ch = 640, n_head = 10, d_head = 64, depth=2, context_dim=2048
        output_blocks_0_1.in_channels = 4 * model_channels;
        output_blocks_0_1.n_head = 20;
        output_blocks_0_1.d_head = num_head_channels; // 64
        output_blocks_0_1.depth = 10;
        output_blocks_0_1.context_dim = context_dim; // 2048
        output_blocks_0_1.create_weight_tensors(ctx);

        output_blocks_1_1.in_channels = 4 * model_channels;
        output_blocks_1_1.n_head = 20;
        output_blocks_1_1.d_head = num_head_channels; // 64
        output_blocks_1_1.depth = 10;
        output_blocks_1_1.context_dim = context_dim; // 2048
        output_blocks_1_1.create_weight_tensors(ctx);

        output_blocks_2_1.in_channels = 4 * model_channels;
        output_blocks_2_1.n_head = 20;
        output_blocks_2_1.d_head = num_head_channels; // 64
        output_blocks_2_1.depth = 10;
        output_blocks_2_1.context_dim = context_dim; // 2048
        output_blocks_2_1.create_weight_tensors(ctx);

        output_blocks_3_1.in_channels = 2 * model_channels;
        output_blocks_3_1.n_head = 10;
        output_blocks_3_1.d_head = num_head_channels; // 64
        output_blocks_3_1.depth = 2;
        output_blocks_3_1.context_dim = context_dim; // 2048
        output_blocks_3_1.create_weight_tensors(ctx);

        output_blocks_4_1.in_channels = 2 * model_channels;
        output_blocks_4_1.n_head = 10;
        output_blocks_4_1.d_head = num_head_channels; // 64
        output_blocks_4_1.depth = 2;
        output_blocks_4_1.context_dim = context_dim; // 2048
        output_blocks_4_1.create_weight_tensors(ctx);

        output_blocks_5_1.in_channels = 2 * model_channels;
        output_blocks_5_1.n_head = 10;
        output_blocks_5_1.d_head = num_head_channels; // 64
        output_blocks_5_1.depth = 2;
        output_blocks_5_1.context_dim = context_dim; // 2048
        output_blocks_5_1.create_weight_tensors(ctx);

        // UpSampleBlock
        // name = output_blocks.2.2, ch = 1280
        // name = output_blocks.5.2, ch = 640
        output_blocks_2_2.channels = 4 * model_channels;
        output_blocks_2_2.out_channels = 4 * model_channels;
        output_blocks_2_2.create_weight_tensors(ctx);

        output_blocks_5_2.channels = 2 * model_channels;
        output_blocks_5_2.out_channels = 2 * model_channels;
        output_blocks_5_2.create_weight_tensors(ctx);


        // Linear
        time_embed_0.in_features = model_channels;
        time_embed_0.out_features = time_embed_dim;
        time_embed_0.create_weight_tensors(ctx);
        time_embed_2.in_features = time_embed_dim;
        time_embed_2.out_features = time_embed_dim;
        time_embed_2.create_weight_tensors(ctx);

        // Linear
        label_emb_0_0.in_features = adm_in_channels;
        label_emb_0_0.out_features = time_embed_dim;
        label_emb_0_0.create_weight_tensors(ctx);
        label_emb_0_2.in_features = time_embed_dim;
        label_emb_0_2.out_features = time_embed_dim;
        label_emb_0_2.create_weight_tensors(ctx);

        // GroupNorm32
        out_0.num_channels = model_channels;
        out_0.create_weight_tensors(ctx);

        // Conv2d
        out_2.in_channels = model_channels;
        out_2.out_channels = out_channels;
        out_2.kernel_size = {3, 3};
        out_2.padding = {1, 1};
        out_2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[1024];
        snprintf(s, sizeof(s), "%s%s", prefix, "input_blocks.0.0.");
        input_blocks_0_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "input_blocks.1.0.");
        input_blocks_1_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "input_blocks.2.0.");
        input_blocks_2_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "input_blocks.4.0.");
        input_blocks_4_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "input_blocks.5.0.");
        input_blocks_5_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "input_blocks.7.0.");
        input_blocks_7_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "input_blocks.8.0.");
        input_blocks_8_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "input_blocks.4.1.");
        input_blocks_4_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "input_blocks.5.1.");
        input_blocks_5_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "input_blocks.7.1.");
        input_blocks_7_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "input_blocks.8.1.");
        input_blocks_8_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "input_blocks.3.0.");
        input_blocks_3_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "input_blocks.6.0.");
        input_blocks_6_0.setup_weight_names(s);

        // middle block
        snprintf(s, sizeof(s), "%s%s", prefix, "middle_block_0.");
        middle_block_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "middle_block_1.");
        middle_block_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "middle_block_2.");
        middle_block_2.setup_weight_names(s);


        // output blocks
        snprintf(s, sizeof(s), "%s%s", prefix, "output_blocks.0.0.");
        output_blocks_0_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "output_blocks.1.0.");
        output_blocks_1_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "output_blocks.2.0.");
        output_blocks_2_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "output_blocks.3.0.");
        output_blocks_3_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "output_blocks.3.0.");
        output_blocks_4_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "output_blocks.5.0.");
        output_blocks_5_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "output_blocks.6.0.");
        output_blocks_6_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "output_blocks.7.0.");
        output_blocks_7_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "output_blocks.8.0.");
        output_blocks_8_0.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "output_blocks.0.1.");
        output_blocks_0_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "output_blocks.1.1.");
        output_blocks_1_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "output_blocks.2.1.");
        output_blocks_2_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "output_blocks.3.1.");
        output_blocks_3_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "output_blocks.4.1.");
        output_blocks_4_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "output_blocks.5.1.");
        output_blocks_5_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "output_blocks.2.2.");
        output_blocks_2_2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "output_blocks.5.2.");
        output_blocks_5_2.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "time_embed.0.");
        time_embed_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "time_embed.2.");
        time_embed_2.setup_weight_names(s);

        // Linear
        snprintf(s, sizeof(s), "%s%s", prefix, "label_emb.0.0.");
        label_emb_0_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "label_emb.0.2.");
        label_emb_0_2.setup_weight_names(s);

        // out_0/out_2
        snprintf(s, sizeof(s), "%s%s", prefix, "out.0.");
        out_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "out.2.");
        out_2.setup_weight_names(s);

        CheckPoint("-------------------- OK");
    }


    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* timesteps,
                                struct ggml_tensor* context,
                                struct ggml_tensor* y = NULL) {
        if (context != NULL && context->ne[2] != x->ne[3]) {
            context = ggml_repeat(ctx, context, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, context->ne[0], context->ne[1], x->ne[3]));
        }
        if (y != NULL && y->ne[1] != x->ne[3]) {
            y = ggml_repeat(ctx, y, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, y->ne[0], x->ne[3]));
        }

        auto t_emb = ggml_nn_timestep_embedding(ctx, timesteps, model_channels);  // [N, model_channels]

        auto emb = time_embed_0.forward(ctx, t_emb);
        emb = ggml_silu_inplace(ctx, emb);
        emb = time_embed_2.forward(ctx, emb);  // [N, time_embed_dim]

        if (y != NULL) {
            auto label_emb = label_emb_0_0.forward(ctx, y);
            label_emb = ggml_silu_inplace(ctx, label_emb);
            label_emb = label_emb_0_2.forward(ctx, label_emb);  // [N, time_embed_dim]
            emb = ggml_add(ctx, emb, label_emb);  // [N, time_embed_dim]
        }

        // input_blocks
        std::vector<struct ggml_tensor*> hs;

        // input block 0
        auto h = input_blocks_0_0.forward(ctx, x);
        // ggml_set_name(h, "bench-start");
        hs.push_back(h);

        // input block 1-11
        size_t len_mults = channel_mult.size();
        // [N, 4*model_channels, h/8, w/8]

        // i == 0
        h = input_blocks_1_0.forward(ctx, h, emb);
        hs.push_back(h);
        h = input_blocks_2_0.forward(ctx, h, emb);
        hs.push_back(h);
        h = input_blocks_3_0.forward(ctx, h);
        hs.push_back(h);

        // i == 1
        h = input_blocks_4_0.forward(ctx, h, emb);
        hs.push_back(h);
        h = input_blocks_5_0.forward(ctx, h, emb);
        hs.push_back(h);
        h = input_blocks_6_0.forward(ctx, h);
        hs.push_back(h);
        
        // i == 2
        h = input_blocks_7_0.forward(ctx, h, emb);
        h = input_blocks_7_1.forward(ctx, h, context);
        hs.push_back(h);
        h = input_blocks_8_0.forward(ctx, h, emb);
        h = input_blocks_8_1.forward(ctx, h, context);
        hs.push_back(h);


        // middle_block
        h = middle_block_0.forward(ctx, h, emb);       // [N, 4*model_channels, h/8, w/8]
        h = middle_block_1.forward(ctx, h, context);   // [N, 4*model_channels, h/8, w/8]
        h = middle_block_2.forward(ctx, h, emb);       // [N, 4*model_channels, h/8, w/8]

        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[controls.size() - 1], control_strength);
            h  = ggml_add(ctx, h, cs);  // middle control
        }
        int control_offset = controls.size() - 2;

        // output_blocks
        // case i == 2
        auto h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);
        h = output_blocks_0_0.forward(ctx, h, emb); // output_blocks_0_0
        h = output_blocks_0_1.forward(ctx, h, context);
        h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);
        h = output_blocks_1_0.forward(ctx, h, emb); // output_blocks_1_0
        h = output_blocks_1_1.forward(ctx, h, context);
        h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);        
        h = output_blocks_2_0.forward(ctx, h, emb); // output_blocks_2_0
        h = output_blocks_2_1.forward(ctx, h, context);
        h = output_blocks_2_2.forward(ctx, h);

        // case i == 1
        h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);        
        h = output_blocks_3_0.forward(ctx, h, emb); // output_blocks_3_0
        h = output_blocks_3_1.forward(ctx, h, context);
        h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);        
        h = output_blocks_4_0.forward(ctx, h, emb); // output_blocks_4_0
        h = output_blocks_4_1.forward(ctx, h, context);
        h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);        
        h = output_blocks_5_0.forward(ctx, h, emb);  // output_blocks_5_0
        h = output_blocks_5_1.forward(ctx, h, context);
        h = output_blocks_5_2.forward(ctx, h);

        // case i == 0
        h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);        
        h = output_blocks_6_0.forward(ctx, h, emb); // output_blocks_6_0
        h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);        
        h = output_blocks_7_0.forward(ctx, h, emb); //output_blocks_7_0
        h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);        
        h = output_blocks_8_0.forward(ctx, h, emb); // output_blocks_8_0


        // out
        h = out_0.forward(ctx, h);
        h = ggml_silu_inplace(ctx, h);
        h = out_2.forward(ctx, h);
        // ggml_set_name(h, "bench-end");
        return h;  // [N, out_channels, h, w]
    }

};


#endif  // __UNET_HPP__