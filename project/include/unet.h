#ifndef __UNET_HPP__
#define __UNET_HPP__

#include "ggml_engine.h"
#include "ggml_nn.h"

/*==================================================== UnetModel =====================================================*/

struct GEGLU {
    int64_t dim_in;
    int64_t dim_out;

    // network params
    struct Linear proj;


    void create_weight_tensors(struct ggml_context* ctx) {
        proj.create_weight_tensors(ctx);
    }

    void setup_weight_names(char *prefix) {
        char s[512];
        snprintf(s, sizeof(s), "%s%s", prefix, "proj.");
        proj.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [ne3, ne2, ne1, dim_in]
        // return: [ne3, ne2, ne1, dim_out]
        struct ggml_tensor* w = proj.proj;
        struct ggml_tensor* b = proj.bias;

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


class FeedForward {
    int64_t dim;
    int64_t dim_out;

    struct GEGLU net_0;
    struct Linear net_2;

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

    void setup_weight_names(char *prefix) {
        char s[512];
        snprintf(s, sizeof(s), "%s%s", prefix, "net.0.");
        net_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "net.2.");
        net_2.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [ne3, ne2, ne1, dim]
        // return: [ne3, ne2, ne1, dim_out]
        x = net_0->forward(ctx, x);  // [ne3, ne2, ne1, inner_dim]
        x = net_2->forward(ctx, x);  // [ne3, ne2, ne1, dim_out]
        return x;
    }
};

struct CrossAttention {
    int64_t query_dim;
    int64_t context_dim;
    int64_t n_head = 20;
    int64_t d_head = 64;

    struct Linear to_q;
    struct Linear to_k;
    struct Linear to_v;
    struct Linear to_out_0;

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

    void setup_weight_names(char *prefix) {
        char s[512];
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

        auto q = to_q->forward(ctx, x);                                 // [N, n_token, inner_dim]
        q      = ggml_reshape_4d(ctx, q, d_head, n_head, n_token, n);   // [N, n_token, n_head, d_head]
        q      = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));      // [N, n_head, n_token, d_head]
        q      = ggml_reshape_3d(ctx, q, d_head, n_token, n_head * n);  // [N * n_head, n_token, d_head]

        auto k = to_k->forward(ctx, context);                             // [N, n_context, inner_dim]
        k      = ggml_reshape_4d(ctx, k, d_head, n_head, n_context, n);   // [N, n_context, n_head, d_head]
        k      = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));        // [N, n_head, n_context, d_head]
        k      = ggml_reshape_3d(ctx, k, d_head, n_context, n_head * n);  // [N * n_head, n_context, d_head]

        auto v = to_v->forward(ctx, context);                             // [N, n_context, inner_dim]
        v      = ggml_reshape_4d(ctx, v, d_head, n_head, n_context, n);   // [N, n_context, n_head, d_head]
        v      = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));        // [N, n_head, d_head, n_context]
        v      = ggml_reshape_3d(ctx, v, n_context, d_head, n_head * n);  // [N * n_head, d_head, n_context]

        auto kqv = ggml_nn_attention(ctx, q, k, v, false);  // [N * n_head, n_token, d_head]
        kqv      = ggml_reshape_4d(ctx, kqv, d_head, n_token, n_head, n);
        kqv      = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // [N, n_token, n_head, d_head]

        x = ggml_reshape_3d(ctx, kqv, d_head * n_head, n_token, n);  // [N, n_token, inner_dim]

        x = to_out_0->forward(ctx, x);  // [N, n_token, query_dim]
        return x;
    }
};


struct BasicTransformerBlock {
    int64_t dim;
    int64_t n_head = 20;
    int64_t d_head = 64;
    int64_t context_dim;
    bool ff_in_flag = false;

    struct CrossAttention attn1;
    struct FeedForward ff;
    struct CrossAttention attn2;

    struct LayerNorm norm1;
    struct LayerNorm norm2;
    struct LayerNorm norm3;

    // struct LayerNorm norm_in;
    // struct FeedForward ff_in;

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

    void setup_weight_names(char *prefix) {
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
        //     x           = norm_in->forward(ctx, x);
        //     x           = ff_in->forward(ctx, x);
        //     // self.is_res is always True
        //     x = ggml_add(ctx, x, x_skip);
        // }

        auto r = x;
        x      = norm1->forward(ctx, x);
        x      = attn1->forward(ctx, x, x);  // self-attention
        x      = ggml_add(ctx, x, r);
        r      = x;
        x      = norm2->forward(ctx, x);
        x      = attn2->forward(ctx, x, context);  // cross-attention
        x      = ggml_add(ctx, x, r);
        r      = x;
        x      = norm3->forward(ctx, x);
        x      = ff->forward(ctx, x);
        x      = ggml_add(ctx, x, r);

        return x;
    }
};

class SpatialTransformer {
    int64_t in_channels;  // mult * model_channels
    int64_t n_head;
    int64_t d_head;
    int64_t depth       = 1;    // 1
    int64_t context_dim = 768;  // hidden_size, 1024 for VERSION_2_x

    struct GroupNorm32 norm;
    struct Conv2d proj_in;
    struct BasicTransformerBlock transformer_blocks[2];
    struct Conv2d proj_out;

    void create_weight_tensors(struct ggml_context* ctx) {
        int64_t inner_dim = n_head * d_head;  // in_channels

        norm.normalized_shape = in_channels;
        norm.create_weight_tensors(ctx);

        proj_in.in_features = in_channels;
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

        proj_out.in_features = inner_dim;
        proj_out.out_channels = in_channels;
        proj_out.kernel_size = {1, 1};
        proj_out.create_weight_tensors(ctx);
    }

    void setup_weight_names(char *prefix) {
        char s[1024];

        snprintf(s, sizeof(s), "%s%s", prefix, "norm.");
        norm.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "proj_in.");
        proj_in.setup_weight_names(s);

        for (int i = 0; i < depth; i++) {
            snprintf(s, sizeof(s), "%s%s", prefix, "transformer_blocks.%d.", i);
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

        x = norm->forward(ctx, x);
        x = proj_in->forward(ctx, x);  // [N, inner_dim, h, w]

        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));  // [N, h, w, inner_dim]
        x = ggml_reshape_3d(ctx, x, inner_dim, w * h, n);      // [N, h * w, inner_dim]

        for (int i = 0; i < depth; i++) {
            // std::string name       = "transformer_blocks." + std::to_string(i);
            auto transformer_block = transformer_blocks[i]; // std::dynamic_pointer_cast<BasicTransformerBlock>(blocks[name]);
            x = transformer_block->forward(ctx, x, context);
        }

        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));  // [N, inner_dim, h * w]
        x = ggml_reshape_4d(ctx, x, w, h, inner_dim, n);       // [N, inner_dim, h, w]

        // proj_out
        x = proj_out->forward(ctx, x);  // [N, in_channels, h, w]

        x = ggml_add(ctx, x, x_in);
        return x;
    }
};




// ldm.modules.diffusionmodules.openaimodel.UNetModel
class UnetModelBlock {
    int in_channels                        = 4;
    int out_channels                       = 4;
    int num_res_blocks                     = 2;
    std::vector<int> attention_resolutions = {4, 2};
    std::vector<int> channel_mult          = {1, 2, 4};
    std::vector<int> transformer_depth     = {1, 2, 10};
    int time_embed_dim                     = 1280;  // model_channels*4
    int num_heads                          = -1;
    int num_head_channels                  = 64;   // channels // num_heads
    int context_dim                        = 2048;  // 1024 for VERSION_2_x, 2048 for VERSION_XL
    int model_channels  = 320;
    int adm_in_channels = 2816;  // only for VERSION_XL/SVD

    // input_blocks.0 -- input_blocks.8;

    Linear time_embed_0;
    Linear time_embed_2;

    // middle_block.0 -- middle_block.2 

    // out.0, out.2

    // output_blocks.0 -- output_blocks.8
    Linear label_emb_0_0;
    Linear label_emb_0_2;



    size_t get_graph_size()
    {
        return GGML_DEFAULT_GRAPH_SIZE * 5; // 2048 * 5
    }


    UnetModelBlock()  {

        blocks["time_embed.0"] = std::shared_ptr<GGMLBlock>(new Linear(model_channels, time_embed_dim));
        // time_embed_1 is nn.SiLU()
        blocks["time_embed.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim));

        blocks["label_emb.0.0"] = std::shared_ptr<GGMLBlock>(new Linear(adm_in_channels, time_embed_dim));
        // label_emb_1 is nn.SiLU()
        blocks["label_emb.0.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim));

        // input_blocks
        blocks["input_blocks.0.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, model_channels, {3, 3}, {1, 1}, {1, 1}));

        std::vector<int> input_block_chans;
        input_block_chans.push_back(model_channels);
        int ch              = model_channels;
        int input_block_idx = 0;
        int ds              = 1;

        auto get_resblock = [&](int64_t channels, int64_t emb_channels, int64_t out_channels) -> ResBlock* {
            return new ResBlock(channels, emb_channels, out_channels);
        };

        auto get_attention_layer = [&](int64_t in_channels,
                                       int64_t n_head,
                                       int64_t d_head,
                                       int64_t depth,
                                       int64_t context_dim) -> SpatialTransformer* {
            return new SpatialTransformer(in_channels, n_head, d_head, depth, context_dim);
        };

        size_t len_mults = channel_mult.size();
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, mult * model_channels));

                ch = mult * model_channels;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) {
                        d_head = num_head_channels;
                        n_head = ch / d_head;
                    }
                    std::string name = "input_blocks." + std::to_string(input_block_idx) + ".1";
                    blocks[name]     = std::shared_ptr<GGMLBlock>(get_attention_layer(ch,
                                                                                      n_head,
                                                                                      d_head,
                                                                                      transformer_depth[i],
                                                                                      context_dim));
                }
                input_block_chans.push_back(ch);
            }
            if (i != len_mults - 1) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(new DownSampleBlock(ch, ch, false));

                input_block_chans.push_back(ch);
                ds *= 2;
            }
        }

        // middle blocks
        int n_head = num_heads;
        int d_head = ch / num_heads;
        if (num_head_channels != -1) {
            d_head = num_head_channels;
            n_head = ch / d_head;
        }
        blocks["middle_block.0"] = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, ch));
        blocks["middle_block.1"] = std::shared_ptr<GGMLBlock>(get_attention_layer(ch,
                                                                                  n_head,
                                                                                  d_head,
                                                                                  transformer_depth[transformer_depth.size() - 1],
                                                                                  context_dim));
        blocks["middle_block.2"] = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, ch));

        // output_blocks
        int output_block_idx = 0;
        for (int i = (int)len_mults - 1; i >= 0; i--) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks + 1; j++) {
                int ich = input_block_chans.back();
                input_block_chans.pop_back();

                std::string name = "output_blocks." + std::to_string(output_block_idx) + ".0";
                blocks[name]     = std::shared_ptr<GGMLBlock>(get_resblock(ch + ich, time_embed_dim, mult * model_channels));

                ch                = mult * model_channels;
                int up_sample_idx = 1;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) {
                        d_head = num_head_channels;
                        n_head = ch / d_head;
                    }
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + ".1";
                    blocks[name]     = std::shared_ptr<GGMLBlock>(get_attention_layer(ch, n_head, d_head, transformer_depth[i], context_dim));

                    up_sample_idx++;
                }

                if (i > 0 && j == num_res_blocks) {
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + "." + std::to_string(up_sample_idx);
                    blocks[name]     = std::shared_ptr<GGMLBlock>(new UpSampleBlock(ch, ch));

                    ds /= 2;
                }

                output_block_idx += 1;
            }
        }

        // out
        blocks["out.0"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(ch));  // ch == model_channels
        // out_1 is nn.SiLU()
        blocks["out.2"] = std::shared_ptr<GGMLBlock>(new Conv2d(model_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    struct ggml_tensor* resblock_forward(std::string name,
                                         struct ggml_context* ctx,
                                         struct ggml_tensor* x,
                                         struct ggml_tensor* emb) {
        auto block = std::dynamic_pointer_cast<ResBlock>(blocks[name]);
        return block->forward(ctx, x, emb);
    }

    struct ggml_tensor* attention_layer_forward(std::string name,
                                                struct ggml_context* ctx,
                                                struct ggml_tensor* x,
                                                struct ggml_tensor* context) {
        auto block = std::dynamic_pointer_cast<SpatialTransformer>(blocks[name]);
        return block->forward(ctx, x, context);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* x,
                                struct ggml_tensor* timesteps,
                                struct ggml_tensor* context,
                                struct ggml_tensor* y = NULL,
                                std::vector<struct ggml_tensor*> controls = {},
                                float control_strength  = 0.f) {
        if (context != NULL) {
            if (context->ne[2] != x->ne[3]) {
                context = ggml_repeat(ctx, context, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, context->ne[0], context->ne[1], x->ne[3]));
            }
        }

        if (y != NULL) {
            if (y->ne[1] != x->ne[3]) {
                y = ggml_repeat(ctx, y, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, y->ne[0], x->ne[3]));
            }
        }

        auto time_embed_0     = std::dynamic_pointer_cast<Linear>(blocks["time_embed.0"]);
        auto time_embed_2     = std::dynamic_pointer_cast<Linear>(blocks["time_embed.2"]);
        auto input_blocks_0_0 = std::dynamic_pointer_cast<Conv2d>(blocks["input_blocks.0.0"]);

        auto out_0 = std::dynamic_pointer_cast<GroupNorm32>(blocks["out.0"]);
        auto out_2 = std::dynamic_pointer_cast<Conv2d>(blocks["out.2"]);

        auto t_emb = ggml_nn_timestep_embedding(ctx, timesteps, model_channels);  // [N, model_channels]

        auto emb = time_embed_0->forward(ctx, t_emb);
        emb      = ggml_silu_inplace(ctx, emb);
        emb      = time_embed_2->forward(ctx, emb);  // [N, time_embed_dim]

        // SDXL/SVD
        if (y != NULL) {
            auto label_embed_0 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.0"]);
            auto label_embed_2 = std::dynamic_pointer_cast<Linear>(blocks["label_emb.0.2"]);

            auto label_emb = label_embed_0->forward(ctx, y);
            label_emb      = ggml_silu_inplace(ctx, label_emb);
            label_emb      = label_embed_2->forward(ctx, label_emb);  // [N, time_embed_dim]

            emb = ggml_add(ctx, emb, label_emb);  // [N, time_embed_dim]
        } else {
            CheckPoint("y == NULL !");
        }

        // input_blocks
        std::vector<struct ggml_tensor*> hs;

        // input block 0
        auto h = input_blocks_0_0->forward(ctx, x);

        ggml_set_name(h, "bench-start");
        hs.push_back(h);
        // input block 1-11
        size_t len_mults    = channel_mult.size();
        int input_block_idx = 0;
        int ds              = 1;
        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                h                = resblock_forward(name, ctx, h, emb);  // [N, mult*model_channels, h, w]
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    std::string name = "input_blocks." + std::to_string(input_block_idx) + ".1";
                    h                = attention_layer_forward(name, ctx, h, context);  // [N, mult*model_channels, h, w]
                }
                hs.push_back(h);
            }
            if (i != len_mults - 1) {
                ds *= 2;
                input_block_idx += 1;

                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                auto block       = std::dynamic_pointer_cast<DownSampleBlock>(blocks[name]);

                h = block->forward(ctx, h);  // [N, mult*model_channels, h/(2^(i+1)), w/(2^(i+1))]
                hs.push_back(h);
            }
        }
        // [N, 4*model_channels, h/8, w/8]

        // middle_block
        h = resblock_forward("middle_block.0", ctx, h, emb);             // [N, 4*model_channels, h/8, w/8]
        h = attention_layer_forward("middle_block.1", ctx, h, context);  // [N, 4*model_channels, h/8, w/8]
        h = resblock_forward("middle_block.2", ctx, h, emb);             // [N, 4*model_channels, h/8, w/8]

        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[controls.size() - 1], control_strength);
            h       = ggml_add(ctx, h, cs);  // middle control
        }
        int control_offset = controls.size() - 2;

        // output_blocks
        int output_block_idx = 0;
        for (int i = (int)len_mults - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                auto h_skip = hs.back();
                hs.pop_back();

                if (controls.size() > 0) {
                    auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
                    h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
                    control_offset--;
                }

                h = ggml_concat(ctx, h, h_skip, 2);

                std::string name = "output_blocks." + std::to_string(output_block_idx) + ".0";

                h = resblock_forward(name, ctx, h, emb);

                int up_sample_idx = 1;
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + ".1";

                    h = attention_layer_forward(name, ctx, h, context);

                    up_sample_idx++;
                }

                if (i > 0 && j == num_res_blocks) {
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + "." + std::to_string(up_sample_idx);
                    auto block       = std::dynamic_pointer_cast<UpSampleBlock>(blocks[name]);

                    h = block->forward(ctx, h);

                    ds /= 2;
                }

                output_block_idx += 1;
            }
        }

        // out
        h = out_0->forward(ctx, h);
        h = ggml_silu_inplace(ctx, h);
        h = out_2->forward(ctx, h);
        ggml_set_name(h, "bench-end");
        return h;  // [N, out_channels, h, w]
    }
};


#endif  // __UNET_HPP__