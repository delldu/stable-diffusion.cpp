/************************************************************************************
***
*** Copyright 2024 Dell(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Sat 24 Aug 2024 03:22:25 PM CST
***
************************************************************************************/

#ifndef __ADAPTER__H__
#define __ADAPTER__H__
#include "ggml_engine.h"

#pragma GCC diagnostic ignored "-Wformat-truncation"


struct AdapterResnetBlock {
    int channels = 320;

    Conv2d block1;
    Conv2d block2;

    void create_weight_tensors(struct ggml_context* ctx) {
        block1.in_channels = channels;
        block1.out_channels = channels;
        block1.kernel_size = { 3, 3 };
        block1.stride = { 1, 1 };
        block1.padding = { 1, 1 };
        block1.create_weight_tensors(ctx);

        block2.in_channels = channels;
        block2.out_channels = channels;
        block2.kernel_size = { 1, 1 };
        block2.stride = { 1, 1 };
        block2.padding = { 0, 0 };
        block2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "block1.");
        block1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "block2.");
        block2.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // h = self.act(self.block1(x))
        // h = self.block2(h)
        // return h + x
        auto h = block1.forward(ctx, x);
        h = ggml_relu_inplace(ctx, h);
        h = block2.forward(ctx, h);

      	h = ggml_add(ctx, x, h);
        return h;
    }
};

struct AdapterBlock {
    int in_channels = 320;
    int out_channels = 320;
    bool down = false;

    Conv2d in_conv;
    struct AdapterResnetBlock resnets_0;
    struct AdapterResnetBlock resnets_1;

    void create_weight_tensors(struct ggml_context* ctx) {
        if (in_channels != out_channels) {
            in_conv.in_channels = in_channels;
            in_conv.out_channels = out_channels;
            in_conv.kernel_size = { 1, 1 };
            in_conv.stride = { 1, 1 };
            in_conv.padding = { 0, 0 };
            in_conv.create_weight_tensors(ctx);
        }
        
        resnets_0.channels = out_channels;
        resnets_0.create_weight_tensors(ctx);

        resnets_1.channels = out_channels;
        resnets_1.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        if (in_channels != out_channels) {
            snprintf(s, sizeof(s), "%s%s", prefix, "in_conv.");
            in_conv.setup_weight_names(s);
        }

        snprintf(s, sizeof(s), "%s%s", prefix, "resnets.0.");
        resnets_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "resnets.1.");
        resnets_1.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
      if (down) {
          // self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
          x = ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, 2/*k0*/, 2/*k1*/, 2/*s0*/, 2/*s1*/, 0/*p0*/, 0/*p1*/);
      }
      if (in_channels != out_channels) {
        x = in_conv.forward(ctx, x);
      }
      x = resnets_0.forward(ctx, x);
      x = resnets_1.forward(ctx, x);

    	return x;
    }
};

/*
 FullAdapterXL(
  (unshuffle): PixelUnshuffle(downscale_factor=16)
  (conv_in): Conv2d(768, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (body): ModuleList(
    (0): AdapterBlock(
      (downsample): Identity()
      (in_conv): Identity()
      (resnets): Sequential(
        (0): AdapterResnetBlock(
          (block1): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): ReLU()
          (block2): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): AdapterResnetBlock(
          (block1): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): ReLU()
          (block2): Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    (1): AdapterBlock(
      (downsample): Identity()
      (in_conv): Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1))
      (resnets): Sequential(
        (0): AdapterResnetBlock(
          (block1): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): ReLU()
          (block2): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): AdapterResnetBlock(
          (block1): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): ReLU()
          (block2): Conv2d(640, 640, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    (2): AdapterBlock(
      (downsample): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (in_conv): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1))
      (resnets): Sequential(
        (0): AdapterResnetBlock(
          (block1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): ReLU()
          (block2): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): AdapterResnetBlock(
          (block1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): ReLU()
          (block2): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
    (3): AdapterBlock(
      (downsample): Identity()
      (in_conv): Identity()
      (resnets): Sequential(
        (0): AdapterResnetBlock(
          (block1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): ReLU()
          (block2): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): AdapterResnetBlock(
          (block1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (act): ReLU()
          (block2): Conv2d(1280, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
  )
) */

struct FullAdapterXL : GGMLNetwork {
    int R = 16; // downscale_factor = 16;
    int C = 3;
    struct ggml_tensor *a; // pixel unshuffle

    Conv2d conv_in;

    struct AdapterBlock body_0;
    struct AdapterBlock body_1;
    struct AdapterBlock body_2;
    struct AdapterBlock body_3;

    size_t get_graph_size()
    {
        return GGML_DEFAULT_GRAPH_SIZE * 16; // 2048 * 4
    }

    void create_weight_tensors(struct ggml_context* ctx) {
        a = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, R, R, C, C); // downscale_factor == 16

        // (conv_in): Conv2d(768, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv_in.in_channels = 768;
        conv_in.out_channels = 320;
        conv_in.kernel_size = { 3, 3 };
        conv_in.stride = { 1, 1 };
        conv_in.padding = { 1, 1 };
        conv_in.create_weight_tensors(ctx);

        body_0.in_channels = 320;
        body_0.out_channels = 320;
        body_0.create_weight_tensors(ctx);

        body_1.in_channels = 320;
        body_1.out_channels = 640;
        body_1.create_weight_tensors(ctx);

        body_2.in_channels = 640;
        body_2.out_channels = 1280;
        body_2.down = true; // !!!
        body_2.create_weight_tensors(ctx);

        body_3.in_channels = 1280;
        body_3.out_channels = 1280;
        body_3.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        ggml_format_name(a, "net.input_a_for_im2col");

        snprintf(s, sizeof(s), "%s%s", prefix, "conv_in.");
        conv_in.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "body.0.");
        body_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "body.1.");
        body_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "body.2.");
        body_2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "body.3.");
        body_3.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, int argc, struct ggml_tensor* argv[]) {
        struct ggml_tensor* x = argv[0];

        // x = self.unshuffle(x)
        x = ggml_im2col(ctx, a, x, R, R, 0, 0, 1, 1, true, GGML_TYPE_F32);
        x = ggml_permute(ctx, x, 2, 0, 1, 3); // from src index to dst: 0->2, 1->0, 2->1, 3->3
        x = ggml_cont(ctx, x); // import !!!
        // ggml_set_name(x, "unshuffle");
        // ggml_set_output(x);

        x = conv_in.forward(ctx, x);
        // ggml_set_name(x, "conv_in");
        // ggml_set_output(x);

        x = body_0.forward(ctx, x);
        ggml_set_name(x, "body0");
        ggml_set_output(x);

        x = body_1.forward(ctx, x);
        ggml_set_name(x, "body1");
        ggml_set_output(x);

        x = body_2.forward(ctx, x);
        ggml_set_name(x, "body2");
        ggml_set_output(x);

        x = body_3.forward(ctx, x);
        ggml_set_name(x, "body3");
        ggml_set_output(x);

        return x;
    }
};

#endif // __ADAPTER__H__
