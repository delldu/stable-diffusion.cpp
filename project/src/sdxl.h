#ifndef __SDXL_H__
#define __SDXL_H__



struct ModelConfig {
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
};

void print_params(ModelConfig params);
int text2image(ModelConfig params);
int image2image(ModelConfig params);


#endif // __SDXL_H__