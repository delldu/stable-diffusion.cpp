#ifndef __VAE_HPP__
#define __VAE_HPP__

#include "common.hpp"
#include "ggml_extend.hpp"

/*================================================== AutoEncoderKL ===================================================*/

#define VAE_GRAPH_SIZE 20480

class ResnetBlock : public UnaryBlock {
protected:
    int64_t in_channels;
    int64_t out_channels;

public:
    ResnetBlock(int64_t in_channels,
                int64_t out_channels)
        : in_channels(in_channels),
          out_channels(out_channels) {
        // temb_channels is always 0
        blocks["norm1"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(in_channels));
        blocks["conv1"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));

        blocks["norm2"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(out_channels));
        blocks["conv2"] = std::shared_ptr<GGMLBlock>(new Conv2d(out_channels, out_channels, {3, 3}, {1, 1}, {1, 1}));

        if (out_channels != in_channels) {
            blocks["nin_shortcut"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, out_channels, {1, 1}));
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, in_channels, h, w]
        // t_emb is always None
        auto norm1 = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm1"]);
        auto conv1 = std::dynamic_pointer_cast<Conv2d>(blocks["conv1"]);
        auto norm2 = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm2"]);
        auto conv2 = std::dynamic_pointer_cast<Conv2d>(blocks["conv2"]);

        auto h = x;
        h      = norm1->forward(ctx, h);
        h      = ggml_silu_inplace(ctx, h);  // swish
        h      = conv1->forward(ctx, h);
        // return h;

        h = norm2->forward(ctx, h);
        h = ggml_silu_inplace(ctx, h);  // swish
        // dropout, skip for inference
        h = conv2->forward(ctx, h);

        // skip connection
        if (out_channels != in_channels) {
            auto nin_shortcut = std::dynamic_pointer_cast<Conv2d>(blocks["nin_shortcut"]);

            x = nin_shortcut->forward(ctx, x);  // [N, out_channels, h, w]
        }

        h = ggml_add(ctx, h, x);
        return h;  // [N, out_channels, h, w]
    }
};

class AttnBlock : public UnaryBlock {
protected:
    int64_t in_channels;

public:
    AttnBlock(int64_t in_channels)
        : in_channels(in_channels) {
        blocks["norm"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(in_channels));
        blocks["q"]    = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, in_channels, {1, 1}));
        blocks["k"]    = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, in_channels, {1, 1}));
        blocks["v"]    = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, in_channels, {1, 1}));

        blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, in_channels, {1, 1}));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, in_channels, h, w]
        auto norm     = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm"]);
        auto q_proj   = std::dynamic_pointer_cast<Conv2d>(blocks["q"]);
        auto k_proj   = std::dynamic_pointer_cast<Conv2d>(blocks["k"]);
        auto v_proj   = std::dynamic_pointer_cast<Conv2d>(blocks["v"]);
        auto proj_out = std::dynamic_pointer_cast<Conv2d>(blocks["proj_out"]);

        auto h_ = norm->forward(ctx, x);

        const int64_t n = h_->ne[3];
        const int64_t c = h_->ne[2];
        const int64_t h = h_->ne[1];
        const int64_t w = h_->ne[0];

        auto q = q_proj->forward(ctx, h_);                          // [N, in_channels, h, w]
        q      = ggml_cont(ctx, ggml_permute(ctx, q, 1, 2, 0, 3));  // [N, h, w, in_channels]
        q      = ggml_reshape_3d(ctx, q, c, h * w, n);              // [N, h * w, in_channels]

        auto k = k_proj->forward(ctx, h_);                          // [N, in_channels, h, w]
        k      = ggml_cont(ctx, ggml_permute(ctx, k, 1, 2, 0, 3));  // [N, h, w, in_channels]
        k      = ggml_reshape_3d(ctx, k, c, h * w, n);              // [N, h * w, in_channels]

        auto v = v_proj->forward(ctx, h_);              // [N, in_channels, h, w]
        v      = ggml_reshape_3d(ctx, v, h * w, c, n);  // [N, in_channels, h * w]

        h_ = ggml_nn_attention(ctx, q, k, v, false);  // [N, h * w, in_channels]

        h_ = ggml_cont(ctx, ggml_permute(ctx, h_, 1, 0, 2, 3));  // [N, in_channels, h * w]
        h_ = ggml_reshape_4d(ctx, h_, w, h, c, n);               // [N, in_channels, h, w]

        h_ = proj_out->forward(ctx, h_);  // [N, in_channels, h, w]

        h_ = ggml_add(ctx, h_, x);
        return h_;
    }
};


// ldm.modules.diffusionmodules.model.Encoder
class Encoder : public GGMLBlock {
protected:
    int ch                   = 128;
    std::vector<int> ch_mult = {1, 2, 4, 4};
    int num_res_blocks       = 2;
    int in_channels          = 3;
    int z_channels           = 4;
    bool double_z            = true;

public:
    Encoder(int ch,
            std::vector<int> ch_mult,
            int num_res_blocks,
            int in_channels,
            int z_channels,
            bool double_z = true)
        : ch(ch),
          ch_mult(ch_mult),
          num_res_blocks(num_res_blocks),
          in_channels(in_channels),
          z_channels(z_channels),
          double_z(double_z) {
        blocks["conv_in"] = std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, ch, {3, 3}, {1, 1}, {1, 1}));

        size_t num_resolutions = ch_mult.size();

        int block_in = 1;
        for (int i = 0; i < num_resolutions; i++) { // [1, 2, 4, 4]
            if (i == 0) {
                block_in = ch;
            } else {
                block_in = ch * ch_mult[i - 1];
            }
            int block_out = ch * ch_mult[i];
            // down.0, down.1, down.2, down3. 
            for (int j = 0; j < num_res_blocks; j++) { // num_res_blocks
                std::string name = "down." + std::to_string(i) + ".block." + std::to_string(j);
                blocks[name]     = std::shared_ptr<GGMLBlock>(new ResnetBlock(block_in, block_out));
                block_in         = block_out;
            }
            if (i != num_resolutions - 1) {
                std::string name = "down." + std::to_string(i) + ".downsample";
                blocks[name]     = std::shared_ptr<GGMLBlock>(new DownSampleBlock(block_in, block_in, true));
            }
        }

        blocks["mid.block_1"] = std::shared_ptr<GGMLBlock>(new ResnetBlock(block_in, block_in));
        blocks["mid.attn_1"]  = std::shared_ptr<GGMLBlock>(new AttnBlock(block_in));
        blocks["mid.block_2"] = std::shared_ptr<GGMLBlock>(new ResnetBlock(block_in, block_in));

        blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(block_in));
        blocks["conv_out"] = std::shared_ptr<GGMLBlock>(new Conv2d(block_in, double_z ? z_channels * 2 : z_channels, {3, 3}, {1, 1}, {1, 1}));
    }

    virtual struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, in_channels, h, w]

        auto conv_in     = std::dynamic_pointer_cast<Conv2d>(blocks["conv_in"]);
        auto mid_block_1 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_1"]);
        auto mid_attn_1  = std::dynamic_pointer_cast<AttnBlock>(blocks["mid.attn_1"]);
        auto mid_block_2 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_2"]);
        auto norm_out    = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm_out"]);
        auto conv_out    = std::dynamic_pointer_cast<Conv2d>(blocks["conv_out"]);

        auto h = conv_in->forward(ctx, x);  // [N, ch, h, w]

        // downsampling
        size_t num_resolutions = ch_mult.size();
        for (int i = 0; i < num_resolutions; i++) {
            for (int j = 0; j < num_res_blocks; j++) {
                std::string name = "down." + std::to_string(i) + ".block." + std::to_string(j);
                auto down_block  = std::dynamic_pointer_cast<ResnetBlock>(blocks[name]);

                h = down_block->forward(ctx, h);
            }
            if (i != num_resolutions - 1) {
                std::string name = "down." + std::to_string(i) + ".downsample";
                auto down_sample = std::dynamic_pointer_cast<DownSampleBlock>(blocks[name]);

                h = down_sample->forward(ctx, h);
            }
        }

        // middle
        h = mid_block_1->forward(ctx, h);
        h = mid_attn_1->forward(ctx, h);
        h = mid_block_2->forward(ctx, h);  // [N, block_in, h, w]

        // end
        h = norm_out->forward(ctx, h);
        h = ggml_silu_inplace(ctx, h);  // nonlinearity/swish
        h = conv_out->forward(ctx, h);  // [N, z_channels*2, h, w]
        return h;
    }
};

// ldm.modules.diffusionmodules.model.Decoder
class Decoder : public GGMLBlock {
protected:
    int ch                   = 128;
    int out_ch               = 3;
    std::vector<int> ch_mult = {1, 2, 4, 4};
    int num_res_blocks       = 2;
    int z_channels           = 4;
    // bool video_decoder       = false;
    int video_kernel_size    = 3;

    virtual std::shared_ptr<GGMLBlock> get_conv_out(int64_t in_channels,
                                                    int64_t out_channels,
                                                    std::pair<int, int> kernel_size,
                                                    std::pair<int, int> stride  = {1, 1},
                                                    std::pair<int, int> padding = {0, 0}) {
        return std::shared_ptr<GGMLBlock>(new Conv2d(in_channels, out_channels, kernel_size, stride, padding));
    }

    virtual std::shared_ptr<GGMLBlock> get_resnet_block(int64_t in_channels,
                                                        int64_t out_channels) {
        return std::shared_ptr<GGMLBlock>(new ResnetBlock(in_channels, out_channels));
    }

public:
    Decoder(int ch,
            int out_ch,
            std::vector<int> ch_mult,
            int num_res_blocks,
            int z_channels,
            // bool video_decoder    = false,
            int video_kernel_size = 3)
        : ch(ch),
          out_ch(out_ch),
          ch_mult(ch_mult),
          num_res_blocks(num_res_blocks),
          z_channels(z_channels),
          // video_decoder(video_decoder),
          video_kernel_size(video_kernel_size) {
        size_t num_resolutions = ch_mult.size();
        int block_in           = ch * ch_mult[num_resolutions - 1];

        blocks["conv_in"] = std::shared_ptr<GGMLBlock>(new Conv2d(z_channels, block_in, {3, 3}, {1, 1}, {1, 1}));

        blocks["mid.block_1"] = get_resnet_block(block_in, block_in);
        blocks["mid.attn_1"]  = std::shared_ptr<GGMLBlock>(new AttnBlock(block_in));
        blocks["mid.block_2"] = get_resnet_block(block_in, block_in);

        for (int i = num_resolutions - 1; i >= 0; i--) {
            int mult      = ch_mult[i];
            int block_out = ch * mult;
            for (int j = 0; j < num_res_blocks + 1; j++) {
                std::string name = "up." + std::to_string(i) + ".block." + std::to_string(j);
                blocks[name]     = get_resnet_block(block_in, block_out);

                block_in = block_out;
            }
            if (i != 0) {
                std::string name = "up." + std::to_string(i) + ".upsample";
                blocks[name]     = std::shared_ptr<GGMLBlock>(new UpSampleBlock(block_in, block_in));
            }
        }

        blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new GroupNorm32(block_in));
        blocks["conv_out"] = get_conv_out(block_in, out_ch, {3, 3}, {1, 1}, {1, 1});
    }

    virtual struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* z) {
        // z: [N, z_channels, h, w]
        // alpha is always 0
        // merge_strategy is always learned
        // time_mode is always conv-only, so we need to replace conv_out_op/resnet_op to AE3DConv/VideoResBlock
        // AttnVideoBlock will not be used
        auto conv_in     = std::dynamic_pointer_cast<Conv2d>(blocks["conv_in"]);
        auto mid_block_1 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_1"]);
        auto mid_attn_1  = std::dynamic_pointer_cast<AttnBlock>(blocks["mid.attn_1"]);
        auto mid_block_2 = std::dynamic_pointer_cast<ResnetBlock>(blocks["mid.block_2"]);
        auto norm_out    = std::dynamic_pointer_cast<GroupNorm32>(blocks["norm_out"]);
        auto conv_out    = std::dynamic_pointer_cast<Conv2d>(blocks["conv_out"]);

        // conv_in
        auto h = conv_in->forward(ctx, z);  // [N, block_in, h, w]

        // middle
        h = mid_block_1->forward(ctx, h);
        // return h;

        h = mid_attn_1->forward(ctx, h);
        h = mid_block_2->forward(ctx, h);  // [N, block_in, h, w]

        // upsampling
        size_t num_resolutions = ch_mult.size();
        for (int i = num_resolutions - 1; i >= 0; i--) {
            for (int j = 0; j < num_res_blocks + 1; j++) {
                std::string name = "up." + std::to_string(i) + ".block." + std::to_string(j);
                auto up_block    = std::dynamic_pointer_cast<ResnetBlock>(blocks[name]);

                h = up_block->forward(ctx, h);
            }
            if (i != 0) {
                std::string name = "up." + std::to_string(i) + ".upsample";
                auto up_sample   = std::dynamic_pointer_cast<UpSampleBlock>(blocks[name]);

                h = up_sample->forward(ctx, h);
            }
        }

        h = norm_out->forward(ctx, h);
        h = ggml_silu_inplace(ctx, h);  // nonlinearity/swish
        h = conv_out->forward(ctx, h);  // [N, out_ch, h*8, w*8]
        return h;
    }
};

// ldm.models.autoencoder.AutoencoderKL
class AutoencodingEngine : public GGMLBlock {
protected:
    bool decode_only       = true;
    int embed_dim          = 4;
    struct {
        int z_channels           = 4;
        int resolution           = 256;
        int in_channels          = 3;
        int out_ch               = 3;
        int ch                   = 128;
        std::vector<int> ch_mult = {1, 2, 4, 4};
        int num_res_blocks       = 2;
        bool double_z            = true;
    } dd_config;

public:
    AutoencodingEngine(bool decode_only       = true)
        : decode_only(decode_only) {

        blocks["decoder"] = std::shared_ptr<GGMLBlock>(new Decoder(dd_config.ch,
                                                                   dd_config.out_ch,
                                                                   dd_config.ch_mult,
                                                                   dd_config.num_res_blocks,
                                                                   dd_config.z_channels));
        blocks["post_quant_conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(dd_config.z_channels,
                                                                          embed_dim,
                                                                          {1, 1}));
        if (!decode_only) {
            blocks["encoder"] = std::shared_ptr<GGMLBlock>(new Encoder(dd_config.ch,
                                                                       dd_config.ch_mult,
                                                                       dd_config.num_res_blocks,
                                                                       dd_config.in_channels,
                                                                       dd_config.z_channels,
                                                                       dd_config.double_z));
            int factor = dd_config.double_z ? 2 : 1;
            // embed_dim = 4, factor=2, dd_config.z_channels = 4
            blocks["quant_conv"] = std::shared_ptr<GGMLBlock>(new Conv2d(embed_dim * factor,
                                                                         dd_config.z_channels * factor,
                                                                         {1, 1})); // ==> 8, 8, 1, 1
        }
    }

    struct ggml_tensor* decode(struct ggml_context* ctx, struct ggml_tensor* z) {
        // z: [N, z_channels, h, w]
        auto post_quant_conv = std::dynamic_pointer_cast<Conv2d>(blocks["post_quant_conv"]);
        z                    = post_quant_conv->forward(ctx, z);  // [N, z_channels, h, w]
        auto decoder = std::dynamic_pointer_cast<Decoder>(blocks["decoder"]);

        ggml_set_name(z, "bench-start");
        auto h = decoder->forward(ctx, z);
        ggml_set_name(h, "bench-end");
        return h;
    }

    struct ggml_tensor* encode(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, in_channels, h, w]
        auto encoder = std::dynamic_pointer_cast<Encoder>(blocks["encoder"]);

        auto h = encoder->forward(ctx, x);  // [N, 2*z_channels, h/8, w/8]
        auto quant_conv = std::dynamic_pointer_cast<Conv2d>(blocks["quant_conv"]);
        h               = quant_conv->forward(ctx, h);  // [N, 2*embed_dim, h/8, w/8]
        return h;
    }
};

struct AutoEncoderKL : public GGMLModule {
    bool decode_only = true;
    AutoencodingEngine ae;

    AutoEncoderKL(ggml_backend_t backend,
                  ggml_type wtype,
                  bool decode_only       = false)
        : decode_only(decode_only), ae(decode_only), GGMLModule(backend, wtype) {
        ae.init(params_ctx, wtype);
    }

    std::string get_desc() {
        return "vae";
    }

    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        ae.get_param_tensors(tensors, prefix);
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* z, bool decode_graph) {
        struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

        z = to_backend(z);

        struct ggml_tensor* out = decode_graph ? ae.decode(compute_ctx, z) : ae.encode(compute_ctx, z);

        ggml_build_forward_expand(gf, out);

        return gf;
    }

    void compute(const int n_threads,
                 struct ggml_tensor* z,
                 bool decode_graph,
                 struct ggml_tensor** output,
                 struct ggml_context* output_ctx = NULL) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(z, decode_graph);
        };
        // ggml_set_f32(z, 0.5f);
        // print_ggml_tensor(z);
        GGMLModule::compute(get_graph, n_threads, true, output, output_ctx);
    }
};

#endif