/************************************************************************************
***
*** Copyright 2024 Dell(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Sat 24 Aug 2024 03:22:25 PM CST
***
************************************************************************************/


#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "src/sdxl.h"

ModelConfig sdxl_turbo_config;

void help(char* cmd)
{
    printf("Usage: %s [option]\n", cmd);
    printf("    -h, --help                   Display this help\n");

    printf("Input:\n");
    printf("    --device <no>               Set Device (0 -- cpu, 1 -- cuda0, 2 -- cuda1, ..., default: %d)\n", sdxl_turbo_config.device);
    printf("    --model <filename>          Model file (default: %s)\n", sdxl_turbo_config.model_path);
    printf("    --image <filename>          Input image required by image2image\n");
    printf("    --positive <prompt>         Positive prompt\n");
    printf("    --negative <prompt>         Negative prompt (default: %s)\n", sdxl_turbo_config.negative);
    printf("    --noise <strength>          Noise strength (default: %.2f)\n", sdxl_turbo_config.noise_strength);
    printf("    --steps <n>                 Sample steps (default: %d)\n", sdxl_turbo_config.sample_steps);
    printf("    --seed <n>                  Random seed (default: %d)\n", sdxl_turbo_config.seed);

    printf("Control:\n");
    printf("    --control_model <filename>  Contreol model\n");
    printf("    --control_image <filename>  Control image\n");
    printf("    --control_strength <n>      Control strength (default: %.2f)\n", sdxl_turbo_config.control_strength);

    // Output
    printf("Output:\n");
    printf("    --height <n>                Output image height (default: %d)\n", sdxl_turbo_config.height);
    printf("    --width <n>                 Output image width (default: %d)\n", sdxl_turbo_config.width);
    printf("    --output <filename>         Output image filename (default: %s)\n", sdxl_turbo_config.output_path);
    printf("    --verbose                   Verbose output\n");

    exit(1);
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
            sdxl_turbo_config.device = atoi(optarg);
            break;
        case 'm':
            sdxl_turbo_config.model_path = optarg;
            break;
        case 'i':
            sdxl_turbo_config.input_image = optarg;
            break;
        case 'p':
            sdxl_turbo_config.positive = optarg;
            break;
        case 'n':
            sdxl_turbo_config.negative = optarg;
            break;
        case 'N':
            sdxl_turbo_config.noise_strength = atof(optarg);
            break;
        case 's':
            sdxl_turbo_config.sample_steps = atoi(optarg);
            break;
        case 'S':
            sdxl_turbo_config.seed = atoi(optarg);
            break;
        // Control
        case '0':
            sdxl_turbo_config.control_model_path = optarg;
            break;
        case '1':
            sdxl_turbo_config.control_image_path = optarg;
            break;
        case '2':
            sdxl_turbo_config.control_strength = atof(optarg);
            break;
        // Output
        case 'H':
            sdxl_turbo_config.height = atoi(optarg);
            break;
        case 'W':
            sdxl_turbo_config.width = atoi(optarg);
            break;
        case 'O':
            sdxl_turbo_config.output_path = optarg;
            break;
        case 'V':
            sdxl_turbo_config.verbose = 1;
            break;
        case 'h': // help
        default:
            help(argv[0]);
            break;
        }
    }


    if (strlen(sdxl_turbo_config.input_image) > 0) {
        return image2image(&sdxl_turbo_config);
    }

    if (strlen(sdxl_turbo_config.positive) > 0) {
        return text2image(&sdxl_turbo_config);
    }

    // Without input image and positive prompt
    help(argv[0]);
    return -1;
}



