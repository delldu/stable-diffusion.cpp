/************************************************************************************
***
***	Copyright 2024 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, Tue 30 Jan 2024 11:52:34 PM CST
***
************************************************************************************/

/************************************************************************************
Header only file for ggml engine

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h> 
************************************************************************************/

#ifndef _GGML_ENGINE_H
#define _GGML_ENGINE_H

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <nimage/image.h> // IMAGE, TENSOR ...

#include <vector>
#include <unordered_map>
#include <algorithm>

#include "meta_tensor.h"

#ifdef GGML_CUDA
#define GGML_USE_CUDA
#define GGML_USE_CUBLAS
#endif

#ifdef GGML_METAL
#define GGML_USE_METAL
#endif


#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif


#define ENGINE_VERSION "1.0.0"
#define MAX_INPUT_TENSORS 8
#define CheckPoint(fmt, arg...) printf("# CheckPoint: %d(%s): " fmt "\n", (int)__LINE__, __FILE__, ##arg)


// GGML Engine
struct GGMLEngine {
    // General
    int device = 0; // 0 -- CPU, 1 -- CUDA 0, 2 -- CUDA 1, ...
    int cpu_threads = 1;

    // Context
    struct ggml_context* inputs_context = NULL;
    struct ggml_context* weight_context = NULL;

    // Backend
    struct ggml_backend* backend = NULL;
    size_t backend_buffer_size = 0; // ONLY reference backend buffer size, not exactly  ...
    struct ggml_backend_buffer* inputs_backend_buffer = NULL;
    struct ggml_backend_buffer* weight_backend_buffer = NULL;

    // Model weight
    const char* model_name = "";
    const char* weight_prefix = "";

    // Graph
    void *graph_cpu_buffer = NULL;

    // Output tensors
    std::unordered_map<char *, TENSOR *> output_tensors = {};
};


struct GGMLModel {
    std::vector<char *> file_paths;
    std::vector<META_TENSOR *> meta_tensors;

    int preload(const char *model_name);
    void remap(const char *oldkey, const char *newkey);
    void dump();
    void clear();
};

struct GGMLNetwork {
public:
    void dump();
    void set_device(int device) { m_ggml_engine.device = device; }
    bool load(const char* model_path, const char* prefix);
    int load_weight(GGMLModel *model, const char *prefix);

    int start_engine();
    TENSOR* engine_forward(int argc, TENSOR* argv[]);
    TENSOR* get_output_tensor(char *name);
    void stop_engine();

    virtual void create_weight_tensors(struct ggml_context* ctx) = 0;
    virtual void setup_weight_names(const char* prefix) = 0;
    virtual struct ggml_tensor* forward(struct ggml_context* ctx, int argc, struct ggml_tensor* argv[])
    {
        // GGML_UNUSED(ctx);
        GGML_UNUSED(argc);
        auto x = argv[0];
        return ggml_dup_inplace(ctx, x); // do not use 'return x;' directly !!!
    }
    virtual size_t get_graph_size()
    {
        return GGML_DEFAULT_GRAPH_SIZE; // 2048
    }

protected:
    GGMLEngine m_ggml_engine = {};
    int m_network_init();
    struct ggml_cgraph* m_build_graph(int argc, struct ggml_tensor* argv[]);
    TENSOR* m_compute(int argc, struct ggml_tensor* argv[]);
    void m_clear_output_tensors();
};

int set_tensor_value(struct ggml_tensor* tensor, TENSOR* nt, bool to_backend); // nt -- nimage tensor
TENSOR* get_tensor_value(struct ggml_tensor* tensor, bool from_backend);
void ggml_tensor_dump(struct ggml_tensor* tensor);

#endif // _GGML_ENGINE_H

// ----------------------------------------------------------------------------------
#ifdef GGML_ENGINE_IMPLEMENTATION
    #ifdef GGML_METAL
    #define GGML_USE_METAL
    #include "ggml-metal.h"
    #endif

    #include "ggml-cuda.h"

    #include <thread>
    #include <vector>

    #define for_each_context_tensor(ctx)                                                                                   \
        for (struct ggml_tensor* t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t))

    static char* _find_model_path(const char* model_name);
    static bool _same_data_shape(struct ggml_tensor* tensor, TENSOR* nt);

    static struct ggml_backend* _device_backend_init(int device, int* ok_device);
    static int _engine_backend_init(GGMLEngine* eng);
    static bool _backend_is_cpu(struct ggml_backend* backend);

    // --------------------------------------------------------------------------------------
    static bool _backend_is_cpu(struct ggml_backend* backend)
    {
        if (ggml_backend_is_cpu(backend)) {
            return true;
        }
    #ifdef GGML_USE_METAL
        if (ggml_cpu_has_metal() && ggml_backend_is_metal(backend)) {
            return true;
        }
    #endif
        return false;
    }

    int GGMLNetwork::start_engine()
    {
        syslog_info("Start Engine (%s) ...", ENGINE_VERSION);

        GGMLEngine* eng = &m_ggml_engine;

        ggml_time_init(); // Nothing for linux but something on windows
        check_point(m_network_init() == RET_OK);
        check_point(_engine_backend_init(eng) == RET_OK);

        syslog_info("Start Engine OK.");

        return RET_OK;
    }

    void GGMLNetwork::stop_engine()
    {
        GGMLEngine* eng = &m_ggml_engine;

        syslog_info("Stop Engine ...");
        // system("nvidia-smi");
        m_clear_output_tensors();

        // Clean backend
        {
            if (eng->inputs_backend_buffer != NULL)
                ggml_backend_buffer_free(eng->inputs_backend_buffer);
            if (eng->weight_backend_buffer != NULL)
                ggml_backend_buffer_free(eng->weight_backend_buffer);
            if (eng->backend != NULL)
                ggml_backend_free(eng->backend);
        }

        // Clean context
        {
            if (eng->inputs_context != NULL)
                ggml_free(eng->inputs_context);
            if (eng->weight_context != NULL)
                ggml_free(eng->weight_context);
            if (eng->graph_cpu_buffer)
                free(eng->graph_cpu_buffer);
            memset((void*)eng, 0, sizeof(GGMLEngine));
        }
        // system("nvidia-smi");
    }

    int GGMLNetwork::m_network_init()
    {
        int num_tensors;
        GGMLEngine* eng = &m_ggml_engine;

        if (eng->weight_context) // do not repeat init ...
            return RET_OK;

        int64_t start_time = ggml_time_ms();

        // Set default threads
        eng->cpu_threads = std::thread::hardware_concurrency();
        // Get num of tensors and memoy size via temp context for more presion
        {
            // ggml_tensor_overhead() == 400
            struct ggml_init_params params = {
                /*.mem_size   =*/ 16 * 1024 * 1024,  // 16M
                /*.mem_buffer =*/NULL,
                /*.no_alloc   =*/true, // the tensors no need to be allocated later
            };
            struct ggml_context* temp_context = ggml_init(params);
            check_point(temp_context);
            create_weight_tensors(temp_context);

            num_tensors = 0;
            eng->backend_buffer_size = 0;
            for_each_context_tensor(temp_context)
            {
                num_tensors++;
                eng->backend_buffer_size += ggml_nbytes(t);
            }
            ggml_free(temp_context);
        }

        // Create tensor and their memory on device cpu
        {
            size_t mem_size = ggml_tensor_overhead() * num_tensors;

            struct ggml_init_params params = {
                /*.mem_size   =*/mem_size,
                /*.mem_buffer =*/NULL,
                /*.no_alloc   =*/true, // the tensors will be allocated later
            };

            eng->weight_context = ggml_init(params);
            check_point(eng->weight_context != NULL);

            // eng->weight_context != NULL
            {
                create_weight_tensors(eng->weight_context);
                setup_weight_names(""); // prefix_name
            }
        }

        check_point(eng->weight_context != NULL);

        syslog_info("Network initialising spends %ld ms", ggml_time_ms() - start_time);

        return RET_OK;
    }

    static int _engine_backend_init(GGMLEngine* eng)
    {
        check_point(eng->weight_context != NULL);

        // Create backend and backend buffer according to network ...
        {
            eng->backend = _device_backend_init(eng->device, &eng->device);
            check_point(eng->backend != NULL);

            eng->weight_backend_buffer = ggml_backend_alloc_ctx_tensors(eng->weight_context, eng->backend);
            // check_point(eng->weight_backend_buffer != NULL);
        }

        // Set CPU threads ...
        {
            if (ggml_backend_is_cpu(eng->backend)) {
                ggml_backend_cpu_set_n_threads(eng->backend, eng->cpu_threads);
            }
    #ifdef GGML_USE_METAL
            if (ggml_cpu_has_metal() && ggml_backend_is_metal(eng->backend)) {
                ggml_backend_metal_set_n_cb(eng->backend, eng->cpu_threads);
            }
    #endif        
        }

        return RET_OK;
    }

    void GGMLNetwork::dump()
    {
        check_avoid(m_ggml_engine.weight_context);

        syslog_info("Network information: ");
        syslog_info("  CPU threads: %d", m_ggml_engine.cpu_threads);
        syslog_info("  Backend device NO: %d, name: %s, buffer: %.2f MB",
             m_ggml_engine.device, ggml_backend_name(m_ggml_engine.backend),
             m_ggml_engine.backend_buffer_size / (1024.0 * 1024.0));
        syslog_info("  Weight model: [%s], prefix: [%s]", m_ggml_engine.model_name, m_ggml_engine.weight_prefix);

        syslog_info("Network tensors:");
        if (m_ggml_engine.inputs_context != NULL)
        for_each_context_tensor(m_ggml_engine.inputs_context) { ggml_tensor_dump(t); }
        if (m_ggml_engine.weight_context != NULL)
        for_each_context_tensor(m_ggml_engine.weight_context) { ggml_tensor_dump(t); }
    }

    bool GGMLNetwork::load(const char* model_name, const char* prefix)
    {
        // Only check model exists ...
        char* model_filename = _find_model_path(model_name);
        if (model_filename) {
            free(model_filename);

            m_ggml_engine.model_name = model_name;
            m_ggml_engine.weight_prefix = prefix;

            return true;
        }

        return false;
    }


    int GGMLNetwork::load_weight(GGMLModel *model, const char *prefix)
    {
        GGMLEngine* eng = &m_ggml_engine;
        check_point(eng != NULL);

        size_t prefix_len = strlen(prefix);
        bool cpu_backend = _backend_is_cpu(eng->backend);

        int64_t start_time = ggml_time_ms();

        size_t size;
        char temp_name[GGML_MAX_NAME];
        std::vector<char> read_buffer;
        std::vector<char> temp_buffer;
        std::vector<char> cast_buffer;
        std::unordered_map<std::string, META_TENSOR *> temp_meta_maps;
        META_TENSOR *meta, temp_meta;

        for (auto& meta : model->meta_tensors) {
            if (prefix_len > 0 && memcmp(meta->name, prefix, prefix_len) != 0) {
                syslog_debug("Skip '%s' for mismatch '%s' ...", meta->name, prefix);
                continue;
            }
            temp_meta_maps[meta->name] = meta;
        }
        if (temp_meta_maps.empty()) {
            return RET_OK;
        }

        for (size_t file_index = 0; file_index < model->file_paths.size(); file_index++) {
            // open file
            FILE * file = ggml_fopen(model->file_paths[file_index], "rb");
            if (! file) { // skip ?
                syslog_error("Open file '%s'", model->file_paths[file_index]);
                continue;
            }

            for_each_context_tensor(m_ggml_engine.weight_context) {
                snprintf(temp_name, sizeof(temp_name), "%s%s", prefix, t->name);

                auto it = temp_meta_maps.find(temp_name);
                meta = (it == temp_meta_maps.end())? NULL : (META_TENSOR *)it->second;
                if (meta == NULL || meta->file_index != file_index) {
                    syslog_error("'%s' NOT found in model weight.", temp_name);
                    continue;
                }

                // match shape 
                if (meta->ne[0] != t->ne[0] || meta->ne[1] != t->ne[1] || meta->ne[2] != t->ne[2] || meta->ne[3] != t->ne[3]) {
                    syslog_error("%s shape mismatch: got [%ld, %ld, %ld, %ld], expected [%ld, %ld, %ld, %ld]",
                        meta->name, meta->ne[0], meta->ne[1], meta->ne[2], meta->ne[3], t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
                    continue;
                }

                // read ok ?
                size = meta_tensor_nbytes(meta);
                read_buffer.reserve(size);
                fseek(file, meta->offset, SEEK_SET);
                const size_t n = fread(read_buffer.data(), 1, size, file);
                if (n != size) {
                    syslog_error("Read '%s' data from '%s' ...", meta->name, model->file_paths[file_index]);
                    continue;
                }

                if (t->type == meta->type) { // fast set
                    if (cpu_backend) {
                        memcpy(t->data, read_buffer.data(), ggml_nbytes(t));
                    } else {
                        ggml_backend_tensor_set(t, read_buffer.data(), 0, ggml_nbytes(t));
                    }
                } else { // slow set
                    meta->data = read_buffer.data();

                    // Clone temp meta for cast
                    memcpy(&temp_meta, meta, sizeof(META_TENSOR));
                    temp_meta.type = t->type;
                    temp_buffer.reserve(ggml_nbytes(t)); // need add more ?
                    temp_meta.data = temp_buffer.data();

                    // CheckPoint                    
                    // meta_tensor_dump(meta);
                    // // meta_tensor_dump(&temp_meta);
                    // ggml_tensor_dump(t);
                    // printf("----------------------------------------------------\n");
                    check_point(meta_tensor_cast(meta, &temp_meta, cast_buffer) == RET_OK);

                    if (cpu_backend) {
                        memcpy(t->data, temp_meta.data, ggml_nbytes(t));
                    } else {
                        ggml_backend_tensor_set(t, temp_meta.data, 0, ggml_nbytes(t));
                    }
                }
                syslog_debug("Loading %s ... OK", t->name);
            }

            // close file
            fclose(file);
        }
        temp_meta_maps.clear();
        read_buffer.clear();
        temp_buffer.clear();
        cast_buffer.clear();

        syslog_info("Loading weight spends %ld ms", ggml_time_ms() - start_time);
        return true;
    }

    // *******************************************************************************
    // build graph spends less 100us(0.1ms),
    // so we build it every time for dynamic and simple logic
    // *******************************************************************************
    struct ggml_cgraph* GGMLNetwork::m_build_graph(int argc, struct ggml_tensor* argv[])
    {
        CHECK_POINT(argc < MAX_INPUT_TENSORS);
        int64_t start_time = ggml_time_ms();

        size_t buf_size = ggml_tensor_overhead() * this->get_graph_size() + ggml_graph_overhead();
        if (m_ggml_engine.graph_cpu_buffer == NULL) {
            m_ggml_engine.graph_cpu_buffer = malloc(buf_size);
        }
        CHECK_POINT(m_ggml_engine.graph_cpu_buffer != NULL);

        struct ggml_init_params params0 = {
            /*.mem_size   =*/buf_size,
            /*.mem_buffer =*/m_ggml_engine.graph_cpu_buffer, 
            /*.no_alloc   =*/true,
        };

        // Create temp context to build the graph
        struct ggml_context* ctx = ggml_init(params0);
        CHECK_POINT(ctx != NULL);

        // struct ggml_cgraph* gf = ggml_new_graph(ctx);
        // struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 2*GGML_DEFAULT_GRAPH_SIZE, false);
        struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, this->get_graph_size(), false); // !!!!!!!!!
        CHECK_POINT(gf != NULL);


        struct ggml_tensor* result = this->forward(ctx, argc, argv);
        CHECK_POINT(result != NULL);

        // ggml_set_output(result);
        ggml_build_forward_expand(gf, result);
        // ggml_graph_compute_with_ctx(ctx, gf, m_ggml_engine.cpu_threads);

        // Delete temp context
        ggml_free(ctx);
        syslog_info("Building graph spends %ld ms", ggml_time_ms() - start_time);

        return gf;
    }

    void GGMLNetwork::m_clear_output_tensors()
    {
        for (auto& pair : m_ggml_engine.output_tensors) {
            char *n = (char *)pair.first;
            if (n != NULL) {
                free(n);
            }
            TENSOR *t =  (TENSOR *)pair.second;
            if (tensor_valid(t)) {
                tensor_destroy(t);
            }
        }
        m_ggml_engine.output_tensors.clear();
    }


    TENSOR* GGMLNetwork::m_compute(int argc, struct ggml_tensor* argv[])
    {
        TENSOR *output = NULL;

        CHECK_POINT(argc < MAX_INPUT_TENSORS);

        // To support dynamic, we need building graph from scratch ...
        struct ggml_gallocr* compute_gallocr = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(m_ggml_engine.backend));
        CHECK_POINT(compute_gallocr != NULL);

        // Build graph !!!!!!!!!!!!! {
        struct ggml_cgraph* gf = m_build_graph(argc, argv);
        CHECK_POINT(gf != NULL);
        CHECK_POINT(ggml_gallocr_alloc_graph(compute_gallocr, gf)); 
        // Build graph !!!!!!!!!!!!! }

        // Dump compute buffer size ...
        {
            size_t s = ggml_gallocr_get_buffer_size(compute_gallocr, 0);
            syslog_info("Compute backend buffer: %.2f M(%s)", s / 1024.0 / 1024.0, 
                ggml_backend_is_cpu(m_ggml_engine.backend) ? "RAM" : "VRAM");
        }

        // Compute ...
        {
            ggml_backend_graph_compute(m_ggml_engine.backend, gf);

            // ggml_graph_print(gf);
            struct ggml_tensor* y = gf->nodes[gf->n_nodes - 1];
            CHECK_POINT(y != NULL);
            output = get_tensor_value(y, true /*from_backend*/);

            // Save output tensors
            for (int i = 0; i < gf->n_leafs; i++) {
                struct ggml_tensor *leaf = gf->leafs[i];
                if (leaf->flags & GGML_TENSOR_FLAG_OUTPUT) {
                    // Saving leafs ...
                    TENSOR *yt = get_tensor_value(leaf, true /*from_backend*/);
                    m_ggml_engine.output_tensors[strdup(leaf->name)] = yt;
                }
            }
            for (int i = 0; i < gf->n_nodes; i++) {
                struct ggml_tensor *node = gf->nodes[i];
                if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
                    // Saving nodes ...
                    TENSOR *yt = get_tensor_value(node, true /*from_backend*/);
                    m_ggml_engine.output_tensors[strdup(node->name)] = yt;
                }
            }
        }

        ggml_gallocr_free(compute_gallocr);
        return output;
    }

    TENSOR* GGMLNetwork::get_output_tensor(char *name)
    {
        return m_ggml_engine.output_tensors[name];
    }


    TENSOR* GGMLNetwork::engine_forward(int argc, TENSOR* argv[])
    {
        struct ggml_tensor* x[MAX_INPUT_TENSORS];

        CHECK_POINT(argc < MAX_INPUT_TENSORS);

        int64_t start_time = ggml_time_ms();
        m_clear_output_tensors();

        // *******************************************************************************
        // Set input data spends less 2ms,
        // so we build it every time for dynamic and simple logic
        // *******************************************************************************
        // Set user input data
        {
            size_t mem_size = 0;
            for (int i = 0; i < argc; i++) {
                mem_size += argv[i]->batch * argv[i]->chan * argv[i]->height * argv[i]->width * sizeof(float);
            }

            // Re-allocate input backend buffer for dynamic input shape ...
            {
                if (m_ggml_engine.inputs_context != NULL)
                    ggml_free(m_ggml_engine.inputs_context);

                struct ggml_init_params params = {
                    /*.mem_size   =*/ ggml_tensor_overhead() * MAX_INPUT_TENSORS,
                    /*.mem_buffer =*/NULL,
                    /*.no_alloc   =*/true, // the tensors will not be allocated later
                };
                m_ggml_engine.inputs_context = ggml_init(params);
                CHECK_POINT(m_ggml_engine.inputs_context);

                char input_name[64];
                for (int i = 0; i < argc; i++) {
                    snprintf(input_name, sizeof(input_name), "net.input_%d", i);
                    x[i] = ggml_new_tensor_4d(m_ggml_engine.inputs_context, GGML_TYPE_F32, 
                        (int64_t)argv[i]->width, (int64_t)argv[i]->height,
                        (int64_t)argv[i]->chan, (int64_t)argv[i]->batch);
                    ggml_set_name(x[i], input_name);
                }

                // Backend ...
                if (m_ggml_engine.inputs_backend_buffer != NULL)
                    ggml_backend_buffer_free(m_ggml_engine.inputs_backend_buffer);

                m_ggml_engine.inputs_backend_buffer = ggml_backend_alloc_ctx_tensors(
                    m_ggml_engine.inputs_context, m_ggml_engine.backend);
                CHECK_POINT(m_ggml_engine.inputs_backend_buffer != NULL);

                syslog_info("Set input spends %ld ms", ggml_time_ms() - start_time);
            }

            bool cpu_backend = _backend_is_cpu(m_ggml_engine.backend);
            if (cpu_backend) {
                syslog_debug("Set input data to cpu backend ...");
            } else {
                syslog_debug("Set input data to cuda backend ...");
            }

            for (int i = 0; i < argc; i++) {
                if (cpu_backend) {
                    memcpy(x[i]->data, argv[i]->data, ggml_nbytes(x[i]));
                } else {
                    ggml_backend_tensor_set(x[i], argv[i]->data, 0, ggml_nbytes(x[i]));
                }
            }
        }

        // Compute ...
        TENSOR* y = m_compute(argc, x); // y = m_compute(argc, x)

        syslog_info("Engine forward spends %ld ms", ggml_time_ms() - start_time);

        return y;
    }


    static char* _find_model_path(const char* model_name)
    {
        // DO NOT forget to free if return != NULL !!!
        if (access(model_name, F_OK) == 0) {
            syslog_info("Found model '%s'.", model_name);
            return strdup(model_name);
        }

        // Try to find model under modes/
        char filename[512];
        snprintf(filename, sizeof(filename), "models/%s", model_name);
        if (access(filename, F_OK) == 0) {
            syslog_info("Found model '%s'.", filename);
            return strdup(filename);
        }

        syslog_error("Model '%s' NOT Found !!!", model_name);
        return NULL;
    }

    static struct ggml_backend* _device_backend_init(int device, int* ok_device)
    {
        GGML_UNUSED(device);
    #ifdef GGML_USE_CUDA    
        if (device && ggml_cpu_has_cuda()) {
            struct ggml_backend* backend = ggml_backend_cuda_init(device - 1); // cuda 0 ...
            if (backend) {
                syslog_info("Using CUDA(%d) as Backend.", device - 1);
                *ok_device = device;
                return backend;
            }
        }
    #endif

    #ifdef GGML_USE_METAL
        if (ggml_cpu_has_metal()) {
            // ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);
            struct ggml_backend* backend = ggml_backend_metal_init();
            if (backend) {
                syslog_info("Using Metal as Backend.");
                *ok_device = 0; // force set to cpu !!!
                return backend;
            }
        }
    #endif

        // Fallback to CPU backend
        syslog_info("Using CPU as Backend.");
        *ok_device = 0; // force set to cpu !!!
        return ggml_backend_cpu_init();
    }


    void ggml_tensor_dump(struct ggml_tensor* tensor)
    {
        char output_buffer[1024];

        check_avoid(tensor);

        size_t len = 0;
        if (tensor->name) {
            len += snprintf(output_buffer + len, sizeof(output_buffer) - len, "%6s [%6ld,%6ld,%6ld,%6ld], %s", 
                ggml_type_name(tensor->type), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->name);
        } else {
            len += snprintf(output_buffer + len, sizeof(output_buffer) - len, "%6s [%6ld,%6ld,%6ld,%6ld], %s", 
                ggml_type_name(tensor->type), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], "none");
        }

        printf("%s\n", output_buffer);
    }


    static bool _same_data_shape(struct ggml_tensor* tensor, TENSOR* nt)
    {
        // B, C, H, W
        bool ok = (nt->batch == (int)tensor->ne[3] && nt->chan == (int)tensor->ne[2] && nt->height == (int)tensor->ne[1]
            && nt->width == (int)tensor->ne[0]);

        if (!ok) {
            syslog_error("Tensor shape: expect[%d, %d, %d, %d], got [%d, %d, %d, %d]", (int)tensor->ne[3],
                (int)tensor->ne[2], (int)tensor->ne[1], (int)tensor->ne[0], nt->batch, nt->chan, nt->height, nt->width);
            return false;
        }

        return ok;
    }

    TENSOR* get_tensor_value(struct ggml_tensor* tensor, bool from_backend = false)
    {
        CHECK_POINT(tensor);
        void *backend_data = NULL;

        if (from_backend) {
            size_t n = (size_t)ggml_nbytes(tensor);
            backend_data = (void *)malloc(n);
            CHECK_POINT(backend_data != NULL);
            ggml_backend_tensor_get(tensor, backend_data, 0, n);
        }

        // B, C, H, W
        TENSOR* nt = tensor_create((int)tensor->ne[3], (int)tensor->ne[2], (int)tensor->ne[1], (int)tensor->ne[0]);
        CHECK_TENSOR(nt);
        size_t n = nt->batch * nt->chan * nt->height * nt->width;

        if (tensor->type == GGML_TYPE_F32) {
            memcpy(nt->data, from_backend? backend_data : tensor->data, n * sizeof(float));
            if (from_backend)
                free(backend_data);
            return nt;
        }

        if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t*)(from_backend?backend_data : tensor->data), (float*)nt->data, n);
            if (from_backend)
                free(backend_data);
            return nt;
        }

        // Dequantize src data to dst
        auto qtype = ggml_internal_get_type_traits(tensor->type);
        CHECK_POINT(qtype.to_float != NULL);
        qtype.to_float(from_backend?backend_data : tensor->data, (float *)nt->data, n);
        if (from_backend)
            free(backend_data);

        return nt;
    }

    int set_tensor_value(struct ggml_tensor* tensor, TENSOR* nt, bool to_backend = false)
    {
        check_point(tensor != NULL);
        check_tensor(nt);

        // B, C, H, W
        check_point(_same_data_shape(tensor, nt));
        size_t nb = ggml_nbytes(tensor);

        // 1) NOT convert data format ...
        if (tensor->type == GGML_TYPE_F32) {
            if (to_backend) {
                ggml_backend_tensor_set(tensor, nt->data, 0, nb);
            } else {
                memcpy(tensor->data, nt->data, nb);
            }
            return RET_OK;
        }

        // 2) Convert nt->data to tensor->data
        void *dst_data = (void *)malloc(nb);
        check_point(dst_data != NULL);
        if (tensor->type == GGML_TYPE_F16) {
            size_t n = tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
            ggml_fp32_to_fp16_row((float*)nt->data, (ggml_fp16_t*)dst_data, n);
        } else {
            size_t nrows = (size_t)ggml_nrows(tensor);
            size_t n_per_row = ggml_nelements(tensor)/nrows; // tensor->ne[0]
            std::vector<float> matrix(n_per_row, 1.0f);  // dummy importance matrix
            ggml_quantize_chunk(tensor->type, (float *)nt->data, dst_data, 0 /*start*/, nrows, n_per_row, matrix.data());
        }
        if (to_backend) {
            ggml_backend_tensor_set(tensor, dst_data, 0, nb);
        } else {
            memcpy(tensor->data, dst_data, nb);
        }
        free(dst_data);

        return RET_OK;
    }

    int GGMLModel::preload(const char *model_name)
    {
        char *file_name = _find_model_path(model_name);
        check_point(file_name != NULL);

        syslog_info("Preloading weight from '%s' ...", file_name);
        if (std::find(file_paths.begin(), file_paths.end(), file_name) != file_paths.end()) // file_name has been preloaded
            return RET_OK;

        int64_t start_time = ggml_time_ms();

        file_paths.push_back(file_name);
        size_t file_index = file_paths.size() - 1;

        gguf_context* ctx_gguf_ = NULL;
        ggml_context* ctx_meta_ = NULL;
        ctx_gguf_ = gguf_init_from_file(file_name, {true, &ctx_meta_});
        check_point(ctx_gguf_ != NULL);

        int n_tensors = gguf_get_n_tensors(ctx_gguf_);

        size_t total_size  = 0;
        size_t data_offset = gguf_get_data_offset(ctx_gguf_);

        for (int i = 0; i < n_tensors; i++) {
            char *name = gguf_get_tensor_name(ctx_gguf_, i);
            struct ggml_tensor* dummy = ggml_get_tensor(ctx_meta_, name);
            // name === dummy->name !!! strange ...            
            // if (strlen(name) >= GGML_MAX_NAME) {
            //     syslog_info("name = '%s' too long (%ld)", name, strlen(name)); // name === dummy->name !!! strange ...;
            // }

            META_TENSOR *meta = meta_tensor_new(gguf_get_tensor_type(ctx_gguf_, i), ggml_n_dims(dummy), dummy->ne);
            check_point(meta != NULL);

            snprintf(meta->name, sizeof(meta->name), "%s", name);
            // meta->type = gguf_get_tensor_type(ctx_gguf_, i); // dummy->type;
            // meta->n_dims = ggml_n_dims(dummy);
            // for (int j = 0; j < meta->n_dims; j++)
            //     meta->ne[j] = dummy->ne[meta->n_dims - j - 1]; // Reverse 

            meta->offset = data_offset + gguf_get_tensor_offset(ctx_gguf_, i);
            meta->file_index = file_index;

            // if (ggml_nbytes(dummy) != meta_tensor_nbytes(meta)) {
            //     meta_tensor_dump(meta);
            //     CheckPoint("====> name = %s, ggml_nbytes(dummy) = %ld, meta_tensor_nbytes(meta) = %ld ", name, ggml_nbytes(dummy), meta_tensor_nbytes(meta));
            // }
            // check_point(ggml_nbytes(dummy) == meta_tensor_nbytes(meta));

            meta_tensors.push_back(meta); // Save to meta_tensors
        }

        gguf_free(ctx_gguf_);
        ggml_free(ctx_meta_);

        syslog_info("Preloading weight spends %ld ms", ggml_time_ms() - start_time);

        return RET_OK;
    }

   void GGMLModel::remap(const char *oldkey, const char *newkey)
    {
        META_TENSOR *t;
        char *find, buf[2 * GGML_MAX_NAME];
        int klen = strlen(oldkey);

        if (klen == 0) // nothing to do ...
            return;

        for (size_t i = 0; i < meta_tensors.size(); i++) {
            t = meta_tensors[i];
            find = strstr(t->name, oldkey);
            if (find != NULL) {
                *find = '\0';
                snprintf(buf, sizeof(buf), "%s%s%s", t->name, newkey, find + klen);
                snprintf(t->name, sizeof(t->name), "%s", buf);
            }
        }
    }

    void GGMLModel::dump()
    {
        printf("Model Files:\n");
        for (size_t i = 0; i < file_paths.size(); i++) {
            printf("  %ld: %s\n", i, file_paths[i]);
        }

        printf("Model Tensors: %ld\n", meta_tensors.size());
        for (auto& meta : meta_tensors) {
            meta_tensor_dump(meta);
        }
    }

    void GGMLModel::clear()
    {
        for (auto& file : file_paths) {
            free(file);
        }
        file_paths.clear();

        for (auto& meta : meta_tensors) {
            free(meta);
        }
        meta_tensors.clear();
    }


#endif // GGML_ENGINE_IMPLEMENTATION

#define META_TENSOR_IMPLEMENTATION
#include "meta_tensor.h"
#undef META_TENSOR_IMPLEMENTATION

