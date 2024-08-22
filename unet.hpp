#ifndef __UNET_HPP__
#define __UNET_HPP__

#include "common.hpp"
#include "ggml_extend.hpp"
#include "model.h"

/*==================================================== UnetModel =====================================================*/

#define UNET_GRAPH_SIZE 10240


// ldm.modules.diffusionmodules.openaimodel.UNetModel
class UnetModelBlock : public GGMLBlock {
protected:
    SDVersion version = VERSION_1_x;
    // network hparams
    int in_channels                        = 4;
    int out_channels                       = 4;
    int num_res_blocks                     = 2;
    std::vector<int> attention_resolutions = {4, 2, 1};
    std::vector<int> channel_mult          = {1, 2, 4, 4};
    std::vector<int> transformer_depth     = {1, 1, 1, 1};
    int time_embed_dim                     = 1280;  // model_channels*4
    int num_heads                          = 8;
    int num_head_channels                  = -1;   // channels // num_heads
    int context_dim                        = 768;  // 1024 for VERSION_2_x, 2048 for VERSION_XL

public:
    int model_channels  = 320;
    int adm_in_channels = 2816;  // only for VERSION_XL/SVD

    UnetModelBlock(SDVersion version = VERSION_1_x)
        : version(version) {
        if (version == VERSION_2_x) {
            context_dim       = 1024;
            num_head_channels = 64;
            num_heads         = -1;
        } else if (version == VERSION_XL) {
            context_dim           = 2048;
            attention_resolutions = {4, 2};
            channel_mult          = {1, 2, 4};
            transformer_depth     = {1, 2, 10};
            num_head_channels     = 64;
            num_heads             = -1;
        }
        // dims is always 2

        blocks["time_embed.0"] = std::shared_ptr<GGMLBlock>(new Linear(model_channels, time_embed_dim));
        blocks["time_embed.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim));

        if (version == VERSION_XL || version == VERSION_SVD) {
            blocks["label_emb.0.0"] = std::shared_ptr<GGMLBlock>(new Linear(adm_in_channels, time_embed_dim));
            blocks["label_emb.0.2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, time_embed_dim));
        }

        // input_blocks
        blocks["input_blocks.0.0"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, model_channels, {3, 3}, {1, 1}, {1, 1}));

        std::vector<int> input_block_chans;
        input_block_chans.push_back(model_channels);
        int ch = model_channels;
        int input_block_idx = 0;
        int ds = 1;

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
        // CheckPoint("---- len_mults = %ld", len_mults); // {1, 2, 4} ==> 3

        for (int i = 0; i < len_mults; i++) {
            int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) {
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                // name = input_blocks.1.0, ch = 320, time_embed_dim = 1280, mult * model_channels = 320
                // name = input_blocks.2.0, ch = 320, time_embed_dim = 1280, mult * model_channels = 320
                // name = input_blocks.4.0, ch = 320, time_embed_dim = 1280, mult * model_channels = 640
                // name = input_blocks.5.0, ch = 640, time_embed_dim = 1280, mult * model_channels = 640
                // name = input_blocks.7.0, ch = 640, time_embed_dim = 1280, mult * model_channels = 1280
                // name = input_blocks.8.0, ch = 1280, time_embed_dim = 1280, mult * model_channels = 1280

                blocks[name] = std::shared_ptr<GGMLBlock>(get_resblock(ch, time_embed_dim, mult * model_channels));

                ch = mult * model_channels;
                // attention_resolutions = {4, 2}
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) { // ==> 64
                        d_head = num_head_channels; // ==> 64
                        n_head = ch / d_head;
                    }
                    std::string name = "input_blocks." + std::to_string(input_block_idx) + ".1";
                    // channel_mult -- {1, 2, 4}, transformer_depth -- {1, 2, 10}
                    // input_blocks.4.1, ch = 640, n_head = 10, d_head = 64, transformer_depth[i]=2, context_dim = 2048
                    // input_blocks.5.1, ch = 640, n_head = 10, d_head = 64, transformer_depth[i]=2, context_dim = 2048
                    // input_blocks.7.1, ch = 1280, n_head = 20, d_head = 64, transformer_depth[i]=10, context_dim = 2048
                    // input_blocks.8.1, ch = 1280, n_head = 20, d_head = 64, transformer_depth[i]=10, context_dim = 2048

                    blocks[name] = std::shared_ptr<GGMLBlock>(get_attention_layer(ch,
                                                                                      n_head,
                                                                                      d_head,
                                                                                      transformer_depth[i],
                                                                                      context_dim));
                }
                input_block_chans.push_back(ch);
            }
            if (i != len_mults - 1) { // ==> i == 0 || i == 1
                input_block_idx += 1; // ==> 3 | 6
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                // // name = input_blocks.3.0, ch = 320
                // // name = input_blocks.6.0, ch = 640
                blocks[name] = std::shared_ptr<GGMLBlock>(new DownSampleBlock(ch, ch, false));

                input_block_chans.push_back(ch);
                ds *= 2;
            }
        }

        // middle blocks
        int n_head = num_heads;
        int d_head = ch / num_heads;

        // CheckPoint("---- num_head_channels = %ld", num_head_channels); // num_head_channels == 64
        if (num_head_channels != -1) {
            d_head = num_head_channels; // ==> 64
            n_head = ch / d_head;
        }

        // CheckPoint("---- ch = %d, time_embed_dim = %d, n_head = %d, d_head = %d, transformer_depth[transformer_depth.size() - 1] = %d, context_dim=%d",
        //     ch, time_embed_dim, n_head, d_head, transformer_depth[transformer_depth.size() - 1], context_dim);
        // ==> ---- ch = 1280, time_embed_dim = 1280, n_head = 20, d_head = 64, transformer_depth[transformer_depth.size() - 1] = 10, context_dim=2048
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
                // name = output_blocks.0.0, ch + ich = 2560, time_embed_dim = 1280, mult * model_channels=1280
                // name = output_blocks.1.0, ch + ich = 2560, time_embed_dim = 1280, mult * model_channels=1280
                // name = output_blocks.2.0, ch + ich = 1920, time_embed_dim = 1280, mult * model_channels=1280
                // name = output_blocks.3.0, ch + ich = 1920, time_embed_dim = 1280, mult * model_channels=640
                // name = output_blocks.4.0, ch + ich = 1280, time_embed_dim = 1280, mult * model_channels=640
                // name = output_blocks.5.0, ch + ich = 960, time_embed_dim = 1280, mult * model_channels=640
                // name = output_blocks.6.0, ch + ich = 960, time_embed_dim = 1280, mult * model_channels=320
                // name = output_blocks.7.0, ch + ich = 640, time_embed_dim = 1280, mult * model_channels=320
                // name = output_blocks.8.0, ch + ich = 640, time_embed_dim = 1280, mult * model_channels=320
                blocks[name]     = std::shared_ptr<GGMLBlock>(get_resblock(ch + ich, time_embed_dim, mult * model_channels));

                ch = mult * model_channels;
                int up_sample_idx = 1;
                // attention_resolutions = {4, 2}
                if (std::find(attention_resolutions.begin(), attention_resolutions.end(), ds) != attention_resolutions.end()) {
                    int n_head = num_heads;
                    int d_head = ch / num_heads;
                    if (num_head_channels != -1) { // ==> 64
                        d_head = num_head_channels; // ==> 64
                        n_head = ch / d_head;
                    }
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + ".1";
                    // name = output_blocks.0.1, ch = 1280, n_head = 20, d_head = 64, depth=10, context_dim=2048
                    // name = output_blocks.1.1, ch = 1280, n_head = 20, d_head = 64, depth=10, context_dim=2048
                    // name = output_blocks.2.1, ch = 1280, n_head = 20, d_head = 64, depth=10, context_dim=2048
                    // name = output_blocks.3.1, ch = 640, n_head = 10, d_head = 64, depth=2, context_dim=2048
                    // name = output_blocks.4.1, ch = 640, n_head = 10, d_head = 64, depth=2, context_dim=2048
                    // name = output_blocks.5.1, ch = 640, n_head = 10, d_head = 64, depth=2, context_dim=2048
                    blocks[name] = std::shared_ptr<GGMLBlock>(get_attention_layer(ch, n_head, d_head, transformer_depth[i], context_dim));

                    up_sample_idx++;
                }

                if (i > 0 && j == num_res_blocks) {
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + "." + std::to_string(up_sample_idx);
                    // name = output_blocks.2.2, ch = 1280
                    // name = output_blocks.5.2, ch = 640
                    blocks[name] = std::shared_ptr<GGMLBlock>(new UpSampleBlock(ch, ch));

                    ds /= 2;
                }

                output_block_idx += 1;
            }
        }

        // out
        // CheckPoint("---- ch = %d", ch); // ch = 320
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
                                struct ggml_tensor* context,  // cond_latent
                                struct ggml_tensor* y = NULL, // cond_pooled
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
        int ds = 1;
#if 0        
        CheckPoint("--------------------------------------{");
        for (int i = 0; i < len_mults; i++) { // i == 0, 1, 2
            CheckPoint("===> i = %d", i);
            // int mult = channel_mult[i];
            for (int j = 0; j < num_res_blocks; j++) { // j = 0, 1
                input_block_idx += 1;
                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                // name = input_blocks.1.0
                // name = input_blocks.2.0
                // name = input_blocks.4.0
                // name = input_blocks.5.0
                // name = input_blocks.7.0
                // name = input_blocks.8.0
                CheckPoint("h = resblock_forward(\"%s\", ctx, h, emb);", name.c_str());
                h = resblock_forward(name, ctx, h, emb);  // [N, mult*model_channels, h, w]
                // attention_resolutions = {4, 2}, ds = 1, 2, 4(i == 2)
                if (i == 2) {
                    std::string name = "input_blocks." + std::to_string(input_block_idx) + ".1";
                    // name = input_blocks.4.1
                    // name = input_blocks.5.1
                    // name = input_blocks.7.1
                    // name = input_blocks.8.1
                    CheckPoint("h = attention_layer_forward(\"%s\", ctx, h, context);", name.c_str());
                    h  = attention_layer_forward(name, ctx, h, context);  // [N, mult*model_channels, h, w]
                }
                CheckPoint("hs.push_back(h);");
                hs.push_back(h);
            }
            if (i != len_mults - 1) { // i == 0 | 1
                ds *= 2;
                input_block_idx += 1;

                std::string name = "input_blocks." + std::to_string(input_block_idx) + ".0";
                // name = input_blocks.3.0
                // name = input_blocks.6.0
                auto block  = std::dynamic_pointer_cast<DownSampleBlock>(blocks[name]);

                CheckPoint("name = %s, h = block->forward(ctx, h)", name.c_str());
                h = block->forward(ctx, h);  // [N, mult*model_channels, h/(2^(i+1)), w/(2^(i+1))]
                CheckPoint("hs.push_back(h);");
                hs.push_back(h);
            }
        }
        CheckPoint("--------------------------------------} input_block_idx = %d", input_block_idx);
#else
        // i == 0
        h = resblock_forward("input_blocks.1.0", ctx, h, emb);
        hs.push_back(h);
        h = resblock_forward("input_blocks.2.0", ctx, h, emb);
        hs.push_back(h);
        auto block  = std::dynamic_pointer_cast<DownSampleBlock>(blocks["input_blocks.3.0"]);
        h = block->forward(ctx, h);
        hs.push_back(h);

        // i == 1
        h = resblock_forward("input_blocks.4.0", ctx, h, emb);
        hs.push_back(h);
        h = resblock_forward("input_blocks.5.0", ctx, h, emb);
        hs.push_back(h);
        block  = std::dynamic_pointer_cast<DownSampleBlock>(blocks["input_blocks.6.0"]);
        h = block->forward(ctx, h);
        hs.push_back(h);
        
        // i == 2
        h = resblock_forward("input_blocks.7.0", ctx, h, emb);
        h = attention_layer_forward("input_blocks.7.1", ctx, h, context);
        hs.push_back(h);
        h = resblock_forward("input_blocks.8.0", ctx, h, emb);
        h = attention_layer_forward("input_blocks.8.1", ctx, h, context);
        hs.push_back(h);
#endif
        // [N, 4*model_channels, h/8, w/8]

        // middle_block
        h = resblock_forward("middle_block.0", ctx, h, emb);             // [N, 4*model_channels, h/8, w/8]
        h = attention_layer_forward("middle_block.1", ctx, h, context);  // [N, 4*model_channels, h/8, w/8]
        h = resblock_forward("middle_block.2", ctx, h, emb);             // [N, 4*model_channels, h/8, w/8]

        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[controls.size() - 1], control_strength);
            h  = ggml_add(ctx, h, cs);  // middle control
        }
        int control_offset = controls.size() - 2;

#if 0
        CheckPoint("================================================================={");

        // output_blocks
        // CheckPoint(" --- hs.size() = %ld", hs.size()); // hs.size() = 9
        int output_block_idx = 0;
        for (int i = (int)len_mults - 1; i >= 0; i--) { // i = 2, 1, 0
            CheckPoint("// case i == %d", i);

            for (int j = 0; j < num_res_blocks + 1; j++) { // j = 0, 1, 2
                CheckPoint("auto h_skip = hs.back();");        
                auto h_skip = hs.back();
                hs.pop_back();

                if (controls.size() > 0) {
                    auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
                    h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
                    control_offset--;
                }

                h = ggml_concat(ctx, h, h_skip, 2);
                std::string name = "output_blocks." + std::to_string(output_block_idx) + ".0";
                // name = output_blocks.0.0
                // name = output_blocks.1.0
                // name = output_blocks.2.0
                // name = output_blocks.3.0
                // name = output_blocks.4.0
                // name = output_blocks.5.0
                // name = output_blocks.6.0
                // name = output_blocks.7.0
                // name = output_blocks.8.0

                CheckPoint("h = resblock_forward(\"%s\", ctx, h, emb);", name.c_str());        
                h = resblock_forward(name, ctx, h, emb);

                int up_sample_idx = 1;
                // attention_resolutions = {4, 2}, ds = 4, 2, 1
                if (i == 2 || i == 1) {
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + ".1";
                    // name = output_blocks.0.1
                    // name = output_blocks.1.1
                    // name = output_blocks.2.1
                    // name = output_blocks.3.1
                    // name = output_blocks.4.1
                    // name = output_blocks.5.1
                    CheckPoint("h = attention_layer_forward(\"%s\", ctx, h, context);", name.c_str());        
                    h = attention_layer_forward(name, ctx, h, context);
                    up_sample_idx++;
                }

                if (i > 0 && j == num_res_blocks) {
                    std::string name = "output_blocks." + std::to_string(output_block_idx) + "." + std::to_string(up_sample_idx);
                    // name = output_blocks.2.2
                    // name = output_blocks.5.2
                    CheckPoint("h = block->forward(ctx, h); %s", name.c_str());        
                    auto block = std::dynamic_pointer_cast<UpSampleBlock>(blocks[name]);
                    h = block->forward(ctx, h);

                    ds /= 2;
                }

                output_block_idx += 1;
            } // end of j 
        }
        CheckPoint("=================================================================}");
#else
        // case i == 2
        auto h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);
        h = resblock_forward("output_blocks.0.0", ctx, h, emb);
        h = attention_layer_forward("output_blocks.0.1", ctx, h, context);
        h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);
        h = resblock_forward("output_blocks.1.0", ctx, h, emb);
        h = attention_layer_forward("output_blocks.1.1", ctx, h, context);
        h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);        
        h = resblock_forward("output_blocks.2.0", ctx, h, emb);
        h = attention_layer_forward("output_blocks.2.1", ctx, h, context);
        auto block2 = std::dynamic_pointer_cast<UpSampleBlock>(blocks["output_blocks.2.2"]);
        h = block2->forward(ctx, h);

        // case i == 1
        h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);        
        h = resblock_forward("output_blocks.3.0", ctx, h, emb);
        h = attention_layer_forward("output_blocks.3.1", ctx, h, context);
        h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);        
        h = resblock_forward("output_blocks.4.0", ctx, h, emb);
        h = attention_layer_forward("output_blocks.4.1", ctx, h, context);
        h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);        
        h = resblock_forward("output_blocks.5.0", ctx, h, emb);
        h = attention_layer_forward("output_blocks.5.1", ctx, h, context);

        block2 = std::dynamic_pointer_cast<UpSampleBlock>(blocks["output_blocks.5.2"]);
        h = block2->forward(ctx, h);

        // case i == 0
        h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);        
        h = resblock_forward("output_blocks.6.0", ctx, h, emb);
        h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);        
        h = resblock_forward("output_blocks.7.0", ctx, h, emb);
        h_skip = hs.back(); hs.pop_back();
        if (controls.size() > 0) {
            auto cs = ggml_scale_inplace(ctx, controls[control_offset], control_strength);
            h_skip  = ggml_add(ctx, h_skip, cs);  // control net condition
            control_offset--;
        }
        h = ggml_concat(ctx, h, h_skip, 2);        
        h = resblock_forward("output_blocks.8.0", ctx, h, emb);
#endif

        // out
        h = out_0->forward(ctx, h);
        h = ggml_silu_inplace(ctx, h);
        h = out_2->forward(ctx, h);
        ggml_set_name(h, "bench-end");
        return h;  // [N, out_channels, h, w]
    }
};

struct UNetModel : public GGMLModule {
    SDVersion version = VERSION_1_x;
    UnetModelBlock unet;

    UNetModel(ggml_backend_t backend,
              ggml_type wtype,
              SDVersion version = VERSION_1_x)
        : GGMLModule(backend, wtype), unet(version) {
        unet.init(params_ctx, wtype);
    }

    std::string get_desc() {
        return "unet";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        unet.get_param_tensors(tensors, prefix);
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                    struct ggml_tensor* timesteps,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* y                     = NULL,
                                    std::vector<struct ggml_tensor*> controls = {},
                                    float control_strength                    = 0.f) {
        struct ggml_cgraph* gf = ggml_new_graph_custom(compute_ctx, UNET_GRAPH_SIZE, false);

        x         = to_backend(x);
        context   = to_backend(context);
        y         = to_backend(y);
        timesteps = to_backend(timesteps);

        for (int i = 0; i < controls.size(); i++) {
            controls[i] = to_backend(controls[i]);
        }

        struct ggml_tensor* out = unet.forward(compute_ctx,
                                               x,
                                               timesteps,
                                               context,
                                               y,
                                               controls,
                                               control_strength);

        ggml_build_forward_expand(gf, out);

        return gf;
    }

    void compute(int n_threads,
                 struct ggml_tensor* x,
                 struct ggml_tensor* timesteps,
                 struct ggml_tensor* context,
                 // struct ggml_tensor* c_concat,
                 struct ggml_tensor* y,
                 std::vector<struct ggml_tensor*> controls = {},
                 float control_strength                    = 0.f,
                 struct ggml_tensor** output               = NULL,
                 struct ggml_context* output_ctx           = NULL) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(x, timesteps, context, y, controls, control_strength);
        };

        GGMLModule::compute(get_graph, n_threads, false, output, output_ctx);
    }
};

#endif  // __UNET_HPP__