#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "tensor.h"
// #include "include/vae.h"
// #include "include/unet.h"
#include "include/clip.h"

struct ModelParams {
    int device = 1; // 0 -- cpu, 1 -- cuda 0

    // Input ...
    char *model_path = (char *)"/opt/ai_models/sdxl_turbo_q8_0.gguf"; // sdxl_turbo_q8_0.gguf";
    
    char *input_image = (char *)""; 
    char *positive = (char *)"";
    char *negative = (char *)"ugly, deformed, noisy, blurry, NSFW";
    float config_scale   = 1.8f; // CONST !!!
    float noise_strength = 0.75f;
    int sample_steps = 5;
    int seed = -1;

    // Control ...
    char *control_model_path = (char *)"";
    char *control_image_path = (char *)"";
    float control_strength = 0.9f;

    // Output ...
    int width = 512;
    int height = 512;
    char *output_path = (char *)"output.png";

    int verbose = 0;
} sdxl_turbo_params;

void print_params(ModelParams params) {
    printf("Input: \n");
    printf("    device:            %d\n", params.device);
    printf("    model path:        '%s'\n", params.model_path);
    printf("    input image:       '%s'\n", params.input_image);
    printf("    prompt:            '%s'\n", params.positive);
    printf("    negative prompt:   '%s'\n", params.negative);
    printf("    noise strength:    %.2f\n", params.noise_strength);
    printf("    sample steps:      %d\n", params.sample_steps);
    printf("    seed:              %d\n", params.seed);
    printf("Control:\n");
    printf("    control model:     '%s'\n", params.control_model_path);
    printf("    control image:     '%s'\n", params.control_image_path);
    printf("    control strength:  %.2f\n", params.control_strength);
    printf("Output:\n");
    printf("    output height:     %d\n", params.height);
    printf("    output width:      %d\n", params.width);
    printf("    output path:       '%s'\n", params.output_path);
}


void help(char* cmd)
{
    printf("Usage: %s [option]\n", cmd);
    printf("    -h, --help                   Display this help\n");

    printf("Input:\n");
    printf("    --device <no>               Set Device (0 -- cpu, 1 -- cuda0, 2 -- cuda1, ..., default: %d)\n", sdxl_turbo_params.device);
    printf("    --model <filename>          Model file (default: %s)\n", sdxl_turbo_params.model_path);
    printf("    --image <filename>          Input image required by image2image\n");
    printf("    --positive <prompt>         Positive prompt\n");
    printf("    --negative <prompt>         Negative prompt (default: %s)\n", sdxl_turbo_params.negative);
    printf("    --noise <strength>          Noise strength (default: %.2f)\n", sdxl_turbo_params.noise_strength);
    printf("    --steps <n>                 Sample steps (default: %d)\n", sdxl_turbo_params.sample_steps);
    printf("    --seed <n>                  Random seed (default: %d)\n", sdxl_turbo_params.seed);

    printf("Control:\n");
    printf("    --control_model <filename>  Contreol model\n");
    printf("    --control_image <filename>  Control image\n");
    printf("    --control_strength <n>      Control strength (default: %.2f)\n", sdxl_turbo_params.control_strength);

    // Output
    printf("Output:\n");
    printf("    --height <n>                Output image height (default: %d)\n", sdxl_turbo_params.height);
    printf("    --width <n>                 Output image width (default: %d)\n", sdxl_turbo_params.width);
    printf("    --output <filename>         Output image filename (default: %s)\n", sdxl_turbo_params.output_path);
    printf("    --verbose                   Verbose output\n");

    exit(1);
}

int text2image(ModelParams params)
{
    printf("Creating image from text ...\n");

    params.height -= params.height % 64;
    params.width -= params.width % 64;

    print_params(params);

    // AutoEncoderKL net;
    // UNetModel net;
    TextEncoder net;

    net.set_device(params.device);
    // net.load(params.model_path, "first_stage_model.");
    // net.load(params.model_path, "model.diffusion_model.");
    // net.load(params.model_path, "cond_stage_model.");

    net.start_engine();
    net.dump();

    // 
    net.stop_engine();


    return 0;
}

int image2image(ModelParams params)
{
    printf("Creating image from image ...\n");

    params.height -= params.height % 64;
    params.width -= params.width % 64;

    print_params(params);

    return 0;
}


int main(int argc, char** argv)
{
    int optc;
    int option_index = 0;

    struct option long_opts[] = {
         { "help", 0, 0, 'h' },
         // Input ...         
         { "device", 1, 0, 'd' }, 
         { "model", 1, 0, 'm' }, 
         { "image", 1, 0, 'i' }, 
         { "positive", 1, 0, 'p' }, 
         { "negative", 1, 0, 'n' }, 
         { "noise", 1, 0, 'N' }, 
         { "steps", 1, 0, 's' }, 
         { "seed", 1, 0, 'S' }, 

         // Control ...         
         { "control_model", 1, 0, '0' }, 
         { "control_image", 1, 0, '1' }, 
         { "control_strength", 1, 0, '2' }, 

         // Output ...         
         { "height", 1, 0, 'H' }, 
         { "width", 1, 0, 'W' }, 
         { "output", 1, 0, 'O' }, 
         { "verbose", 0, 0, 'V' }, 
         { 0, 0, 0, 0 }
    };

    while ((optc = getopt_long(argc, argv, "h d:m:i:p:n:N:s:S:   0:1:2:  H:W:O:V", long_opts, &option_index)) != EOF) {
        switch (optc) {
        // Input
        case 'd':
            sdxl_turbo_params.device = atoi(optarg);
            break;
        case 'm':
            sdxl_turbo_params.model_path = optarg;
            break;
        case 'i':
            sdxl_turbo_params.input_image = optarg;
            break;
        case 'p':
            sdxl_turbo_params.positive = optarg;
            break;
        case 'n':
            sdxl_turbo_params.negative = optarg;
            break;
        case 'N':
            sdxl_turbo_params.noise_strength = atof(optarg);
            break;
        case 's':
            sdxl_turbo_params.sample_steps = atoi(optarg);
            break;
        case 'S':
            sdxl_turbo_params.seed = atoi(optarg);
            break;
        // Control
        case '0':
            sdxl_turbo_params.control_model_path = optarg;
            break;
        case '1':
            sdxl_turbo_params.control_image_path = optarg;
            break;
        case '2':
            sdxl_turbo_params.control_strength = atof(optarg);
            break;
        // Output
        case 'H':
            sdxl_turbo_params.height = atoi(optarg);
            break;
        case 'W':
            sdxl_turbo_params.width = atoi(optarg);
            break;
        case 'O':
            sdxl_turbo_params.output_path = optarg;
            break;
        case 'V':
            sdxl_turbo_params.verbose = 1;
            break;
        case 'h': // help
        default:
            help(argv[0]);
            break;
        }
    }

    if (sdxl_turbo_params.verbose) {
        print_params(sdxl_turbo_params);
    }

    if (strlen(sdxl_turbo_params.input_image) > 0) {
        return image2image(sdxl_turbo_params);
    }

    if (strlen(sdxl_turbo_params.positive) > 0) {
        return text2image(sdxl_turbo_params);
    }

    // Without input image and positive prompt
    help(argv[0]);
    return -1;
}



