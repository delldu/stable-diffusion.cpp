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

#include <unordered_map>

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
    size_t backend_buffer_size = 0;
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

struct GGMLNetwork {
public:
    void dump();
    void set_device(int device) { m_ggml_engine.device = device; }
    bool load(const char* model_path, const char* prefix);

    bool start_engine();
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
    bool m_network_init();
    struct ggml_cgraph* m_build_graph(int argc, struct ggml_tensor* argv[]);
    TENSOR* m_compute(int argc, struct ggml_tensor* argv[]);
    void m_clear_output_tensors();
};

void *get_cast_data(struct ggml_tensor *x, bool from_backend, ggml_type dtype);
bool set_tensor_value(struct ggml_tensor* tensor, TENSOR* nt, bool to_backend); // nt -- nimage tensor
TENSOR* get_tensor_value(struct ggml_tensor* tensor, bool from_backend);
void dump_ggml_tensor(const char* prefix, struct ggml_tensor* tensor);

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

    static bool _data_type_valid(ggml_type dtype);
    static bool _same_data_shape(struct ggml_tensor* tensor, TENSOR* nt);

    static struct ggml_backend* _device_backend_init(int device, int* ok_device);
    static bool _engine_backend_init(GGMLEngine* eng);
    static bool _backend_is_cpu(struct ggml_backend* backend);
    static bool _load_weight_from_gguf(GGMLEngine* eng);

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

    bool GGMLNetwork::start_engine()
    {
        syslog_info("Start Engine (%s) ...", ENGINE_VERSION);

        GGMLEngine* eng = &m_ggml_engine;

        ggml_time_init(); // Nothing for linux but something on windows
        check_point(m_network_init());
        check_point(_engine_backend_init(eng));
        _load_weight_from_gguf(eng);

        syslog_info("Start Engine OK.");

        return true;
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

    bool GGMLNetwork::m_network_init()
    {
        int num_tensors;
        GGMLEngine* eng = &m_ggml_engine;

        if (eng->weight_context) // do not repeat init ...
            return true;

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
            check_point(eng->weight_context);

            // eng->weight_context != NULL
            {
                create_weight_tensors(eng->weight_context);
                setup_weight_names(""); // prefix_name
            }
        }

        check_point(eng->weight_context);

        syslog_debug("Network initialising spends %ld ms", ggml_time_ms() - start_time);

        return true;
    }

    static bool _engine_backend_init(GGMLEngine* eng)
    {
        check_point(eng->weight_context);

        // Create backend and backend buffer according to network ...
        {
            eng->backend = _device_backend_init(eng->device, &eng->device);
            check_point(eng->backend);

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

        return true;
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
        for_each_context_tensor(m_ggml_engine.inputs_context) { dump_ggml_tensor("  ", t); }
        if (m_ggml_engine.weight_context != NULL)
        for_each_context_tensor(m_ggml_engine.weight_context) { dump_ggml_tensor("  ", t); }
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

    static bool _load_weight_from_gguf(GGMLEngine* eng)
    {
        check_point(eng);
        if (strlen(eng->model_name) < 1) // Skip if no-existed model ...
            return false;

        int64_t start_time = ggml_time_ms();

        char* model_filename = NULL;
        struct gguf_context* ctx_gguf = NULL;
        struct {
            struct ggml_context* context;
        } weight;

        // Loading weight
        {
            syslog_info("Loading weight from '%s' with prefix '%s' ...", eng->model_name, eng->weight_prefix);

            model_filename = _find_model_path(eng->model_name);
            check_point(model_filename);

            struct gguf_init_params params = {
                /*.no_alloc   =*/false,
                /*.ctx        =*/&weight.context,
            };

            ctx_gguf = gguf_init_from_file(model_filename, params);
            if (!ctx_gguf) {
                syslog_error("Loading gguf file '%s'", model_filename);
                free(model_filename);
            }

            check_point(ctx_gguf);
            check_point(weight.context);
        }

        // Network loading weight ...
        {
            size_t prefix_len = strlen(eng->weight_prefix);
            struct ggml_tensor* d = NULL;
            bool cpu_backend = _backend_is_cpu(eng->backend);

            for_each_context_tensor(weight.context)
            {
                if (memcmp(t->name, eng->weight_prefix, prefix_len) != 0) {
                    syslog_debug("Skip '%s' for mismatch '%s' ...", t->name, eng->weight_prefix);
                    continue;
                }

                // Real name should be t->name + prefix_len !!!
                d = ggml_get_tensor(eng->weight_context, t->name + prefix_len);
                if (d == NULL) {
                    syslog_debug("Skip '%s' for not defined in network ...", t->name + prefix_len);
                    continue;
                }
                if (!ggml_are_same_shape(d, t)) {
                    syslog_error("%s shape mismatch: got [%ld, %ld, %ld, %ld], expected [%ld, %ld, %ld, %ld]",
                        d->name, t->ne[0], t->ne[1], t->ne[2], t->ne[3], d->ne[0],
                        d->ne[1], d->ne[2], d->ne[3]);
                    continue;
                }

                // Loading tensors ...
                if (d->type == t->type) { // fast set
                    if (cpu_backend) {
                        memcpy(d->data, t->data, ggml_nbytes(d));
                    } else {
                        ggml_backend_tensor_set(d, t->data, 0, ggml_nbytes(d));
                    }
                } else { // slow set
                    void *temp_data = get_cast_data(t, false /*from_backend */, d->type);
                    check_point(temp_data != NULL);                    

                    if (cpu_backend) {
                        memcpy(d->data, temp_data, ggml_nbytes(d));
                    } else {
                        ggml_backend_tensor_set(d, temp_data, 0, ggml_nbytes(d));
                    }
                    free(temp_data);
                }
                syslog_debug("Loading %s ... OK", d->name);
            }
        }

        // Clean up
        {
            gguf_free(ctx_gguf);
            ggml_free(weight.context);
            free(model_filename);
        }

        syslog_debug("Loading weight spends %ld ms", ggml_time_ms() - start_time);
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
        syslog_debug("Building graph spends %ld ms", ggml_time_ms() - start_time);

        return gf;
    }

    void GGMLNetwork::m_clear_output_tensors()
    {
        for (auto& pair : m_ggml_engine.output_tensors) {
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
                    m_ggml_engine.output_tensors[leaf->name] = yt;
                }
            }
            for (int i = 0; i < gf->n_nodes; i++) {
                struct ggml_tensor *node = gf->nodes[i];
                if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
                    // Saving nodes ...
                    TENSOR *yt = get_tensor_value(node, true /*from_backend*/);
                    m_ggml_engine.output_tensors[node->name] = yt;
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

                syslog_debug("Set input spends %ld ms", ggml_time_ms() - start_time);
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


    void dump_ggml_tensor(const char* prefix, struct ggml_tensor* tensor)
    {
        char output_buffer[1024];

        check_avoid(tensor);

        size_t len = 0;
        if (tensor->name) {
            len += snprintf(output_buffer + len, sizeof(output_buffer) - len, "%s%s: %s, [%ld, %ld, %ld, %ld]", prefix,
                tensor->name, ggml_type_name(tensor->type), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } else {
            len += snprintf(output_buffer + len, sizeof(output_buffer) - len, "%s%s: %s, [%ld, %ld, %ld, %ld]", prefix,
                "none", ggml_type_name(tensor->type), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        }

        syslog_info("%s", output_buffer);
    }

    static bool _data_type_valid(ggml_type dtype)
    {
        return (dtype == GGML_TYPE_I8 || dtype == GGML_TYPE_I16 || dtype == GGML_TYPE_I32 || dtype == GGML_TYPE_F16
            || dtype == GGML_TYPE_F32);
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
        CHECK_POINT(_data_type_valid(tensor->type));

        float *source_data = (float *)get_cast_data(tensor, from_backend, GGML_TYPE_F32);
        CHECK_POINT(source_data);

        // B, C, H, W
        TENSOR* nt = tensor_create((int)tensor->ne[3], (int)tensor->ne[2], (int)tensor->ne[1], (int)tensor->ne[0]);
        if (nt == NULL) {
            free(source_data);
        }
        CHECK_TENSOR(nt);
        size_t n = nt->batch * nt->chan * nt->height * nt->width;
        memcpy(nt->data, source_data, n * sizeof(float)); // Save source data to nt->data

        free(source_data);

        return nt;
    }

    bool set_tensor_value(struct ggml_tensor* tensor, TENSOR* nt, bool to_backend = false)
    {
        check_point(tensor);
        check_tensor(nt);

        // B, C, H, W
        check_point(_same_data_shape(tensor, nt));
        check_point(_data_type_valid(tensor->type));
        size_t nb = ggml_nbytes(tensor);

        // 1) NOT convert data format ...
        if (tensor->type == GGML_TYPE_F32) {
            if (to_backend) {
                ggml_backend_tensor_set(tensor, nt->data, 0, nb);
            } else {
                memcpy(tensor->data, nt->data, nb);
            }
            return true;
        }

        // 2) Convert nt->data to tensor->data
        void *dst_data = (void *)malloc(nb);
        check_point(dst_data != NULL);
        {
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

        return true;
    }

    void *get_cast_data(struct ggml_tensor *x, bool from_backend, ggml_type dtype)
    {
        // Cast type of x to dtype
        void *backend_data = NULL;

        if (from_backend) {
            size_t bn = (size_t)ggml_nbytes(x);

            backend_data = (void *)malloc(bn);
            CHECK_POINT(backend_data != NULL);
            ggml_backend_tensor_get(x, backend_data, 0, bn);
        }

        size_t n = ggml_nelements(x);
        float *float_data = (float *)calloc(n, sizeof(float));
        CHECK_POINT(float_data != NULL);

        // 1) Dequantize data to float_data
        if (x->type == GGML_TYPE_F32) {
            if(from_backend) {
                memcpy((void *)float_data, (void *)backend_data, n * sizeof(float));
                free(backend_data);
            } else {
                memcpy(float_data, x->data, n * sizeof(float));
            }
            return float_data;
        }
        {
            auto qtype = ggml_internal_get_type_traits(x->type);
            CHECK_POINT(qtype.to_float != NULL);
            if (from_backend) {
                qtype.to_float(backend_data, (float *)float_data, n);
                free(backend_data); // backend_data is useless
            } else {
                qtype.to_float(x->data, (float *)float_data, n);
            }
        }

        // 2) Quantize float_data to dst_data
        if (dtype == GGML_TYPE_F32)
            return float_data;

        void *dst_data = (void *)calloc(n, ggml_type_size(dtype));
        if (dst_data == NULL) {
            free(float_data);
        }
        CHECK_POINT(dst_data != NULL);
        {
            size_t nrows = (size_t)ggml_nrows(x);
            size_t n_per_row = n/nrows; // x->ne[0]

            std::vector<float> matrix(n_per_row, 1.0f);  // dummy importance matrix
            ggml_quantize_chunk(dtype, (float *)float_data, dst_data, 0 /*start*/, nrows, n_per_row, matrix.data());
        }

        free(float_data);

        return dst_data; // need free dst_data
    }
#endif // GGML_ENGINE_IMPLEMENTATION

