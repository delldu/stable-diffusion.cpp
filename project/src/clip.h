/************************************************************************************
***
*** Copyright 2024 Dell(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Sat 24 Aug 2024 03:22:25 PM CST
***
************************************************************************************/

#ifndef __CLIP_H__
#define __CLIP_H__

#include <ggml_engine.h>

#include <codecvt>
#include <map>
#include <regex>
#include <set>

enum SDVersion {
    VERSION_1_x,
    VERSION_XL,
};

#pragma GCC diagnostic ignored "-Wformat-truncation"

static std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text);
static std::vector<std::pair<int, std::u32string>> bytes_to_unicode();
static std::u32string utf8_to_utf32(const std::string& utf8_str);
static std::string utf32_to_utf8(const std::u32string& utf32_str);
static std::u32string unicode_value_to_utf32(int unicode_value);

// -----------------------------------------------------------------------------------------------------------------

static std::u32string utf8_to_utf32(const std::string& utf8_str)
{
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.from_bytes(utf8_str);
}

static std::string utf32_to_utf8(const std::u32string& utf32_str)
{
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.to_bytes(utf32_str);
}

static std::u32string unicode_value_to_utf32(int unicode_value)
{
    std::u32string utf32_string = { static_cast<char32_t>(unicode_value) };
    return utf32_string;
}

#if 0
std::pair<std::unordered_map<std::string, float>, std::string> extract_and_remove_lora(std::string text)
{
    std::regex re("<lora:([^:]+):([^>]+)>");
    std::smatch matches;
    std::unordered_map<std::string, float> filename2multiplier;

    while (std::regex_search(text, matches, re)) {
        std::string filename = matches[1].str();
        float multiplier     = std::stof(matches[2].str());

        text = std::regex_replace(text, re, "", std::regex_constants::format_first_only);

        if (multiplier == 0.f) {
            continue;
        }

        if (filename2multiplier.find(filename) == filename2multiplier.end()) {
            filename2multiplier[filename] = multiplier;
        } else {
            filename2multiplier[filename] += multiplier;
        }
    }

    return std::make_pair(filename2multiplier, text);
}
#endif

const std::string BOS_TOKEN = "<|startoftext|>";
const std::string EOS_TOKEN = "<|endoftext|>";
const std::string PAD_TOEKN = "<|endoftext|>";

const int BOS_TOKEN_ID = 49406;
const int EOS_TOKEN_ID = 49407;
const int PAD_TOKEN_ID = 49407;

static std::vector<std::pair<int, std::u32string>> bytes_to_unicode()
{
    std::vector<std::pair<int, std::u32string>> byte_unicode_pairs;
    std::set<int> byte_set;
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    for (int b = 161; b <= 172; ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    for (int b = 174; b <= 255; ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (byte_set.find(b) == byte_set.end()) {
            byte_unicode_pairs.push_back(std::pair<int, std::u32string>(b, unicode_value_to_utf32(n + 256)));
            ++n;
        }
    }
    return byte_unicode_pairs;
}

class CLIPTokenizer {
private:
    SDVersion version = VERSION_1_x;
    std::map<int, std::u32string> byte_encoder;
    std::map<std::u32string, int> byte_decoder;
    std::map<std::u32string, int> encoder;
    std::map<int, std::u32string> decoder;
    std::map<std::pair<std::u32string, std::u32string>, int> bpe_ranks;
    std::regex pat;
    int encoder_len;
    int bpe_len;

    static std::string strip(const std::string& str)
    {
        std::string::size_type start = str.find_first_not_of(" \t\n\r\v\f");
        std::string::size_type end = str.find_last_not_of(" \t\n\r\v\f");

        if (start == std::string::npos) {
            // String contains only whitespace characters
            return "";
        }

        return str.substr(start, end - start + 1);
    }

    static std::string whitespace_clean(std::string text)
    {
        text = std::regex_replace(text, std::regex(R"(\s+)"), " ");
        text = strip(text);
        return text;
    }

    static std::set<std::pair<std::u32string, std::u32string>> get_pairs(const std::vector<std::u32string>& subwords)
    {
        std::set<std::pair<std::u32string, std::u32string>> pairs;
        if (subwords.size() == 0) {
            return pairs;
        }
        std::u32string prev_subword = subwords[0];
        for (int i = 1; i < subwords.size(); i++) {
            std::u32string subword = subwords[i];
            std::pair<std::u32string, std::u32string> pair(prev_subword, subword);
            pairs.insert(pair);
            prev_subword = subword;
        }
        return pairs;
    }

public:
    CLIPTokenizer(SDVersion version = VERSION_1_x)
        : version(version)
    {
        load_from_merges();
    }

    void load_from_merges()
    {
        #include "vocab.h"
        std::string merges_utf8_str(reinterpret_cast<const char*>(merges_utf8_c_str), sizeof(merges_utf8_c_str));
        if (merges_utf8_str.size() == 0) {
            syslog_error("get vocab");
            return;
        }

        auto byte_unicode_pairs = bytes_to_unicode();
        // printf("byte_unicode_pairs have %lu pairs \n", byte_unicode_pairs.size());
        byte_encoder = std::map<int, std::u32string>(byte_unicode_pairs.begin(), byte_unicode_pairs.end());
        for (auto& pair : byte_unicode_pairs) {
            byte_decoder[pair.second] = pair.first;
        }
        std::vector<std::u32string> merges;
        size_t start = 0;
        size_t pos;
        std::u32string merges_utf32_str = utf8_to_utf32(merges_utf8_str);
        while ((pos = merges_utf32_str.find('\n', start)) != std::string::npos) {
            merges.push_back(merges_utf32_str.substr(start, pos - start));
            start = pos + 1;
        }

        GGML_ASSERT(merges.size() == 48895);
        merges = std::vector<std::u32string>(merges.begin() + 1, merges.end());
        std::vector<std::pair<std::u32string, std::u32string>> merge_pairs;
        for (const auto& merge : merges) {
            size_t space_pos = merge.find(' ');
            merge_pairs.emplace_back(merge.substr(0, space_pos), merge.substr(space_pos + 1));
        }
        std::vector<std::u32string> vocab;
        for (const auto& pair : byte_unicode_pairs) {
            vocab.push_back(pair.second);
        }
        for (const auto& pair : byte_unicode_pairs) {
            vocab.push_back(pair.second + utf8_to_utf32("</w>"));
        }
        for (const auto& merge : merge_pairs) {
            vocab.push_back(merge.first + merge.second);
        }
        vocab.push_back(utf8_to_utf32("<|startoftext|>"));
        vocab.push_back(utf8_to_utf32("<|endoftext|>"));

        int i = 0;
        for (const auto& token : vocab) {
            encoder[token] = i;
            decoder[i] = token;
            i++;
        }
        encoder_len = i;

        auto it = encoder.find(utf8_to_utf32("img</w>"));
        // if (it != encoder.end()) {
        //     LOG_DEBUG(" trigger word img already in vocab");
        // } else {
        //     LOG_DEBUG(" trigger word img not in vocab yet");
        // }

        int rank = 0;
        for (const auto& merge : merge_pairs) {
            bpe_ranks[merge] = rank++;
        }
        bpe_len = rank;
    };

    std::u32string bpe(const std::u32string& token)
    {
        std::vector<std::u32string> word;

        for (int i = 0; i < token.size() - 1; i++) {
            word.emplace_back(1, token[i]);
        }
        word.push_back(token.substr(token.size() - 1) + utf8_to_utf32("</w>"));

        std::set<std::pair<std::u32string, std::u32string>> pairs = get_pairs(word);

        if (pairs.empty()) {
            return token + utf8_to_utf32("</w>");
        }

        while (true) {
            auto min_pair_iter = std::min_element(pairs.begin(),
                pairs.end(),
                [&](const std::pair<std::u32string, std::u32string>& a,
                    const std::pair<std::u32string, std::u32string>& b) {
                    if (bpe_ranks.find(a) == bpe_ranks.end()) {
                        return false;
                    } else if (bpe_ranks.find(b) == bpe_ranks.end()) {
                        return true;
                    }
                    return bpe_ranks.at(a) < bpe_ranks.at(b);
                });

            const std::pair<std::u32string, std::u32string>& bigram = *min_pair_iter;

            if (bpe_ranks.find(bigram) == bpe_ranks.end()) {
                break;
            }

            std::u32string first = bigram.first;
            std::u32string second = bigram.second;
            std::vector<std::u32string> new_word;
            int32_t i = 0;

            while (i < word.size()) {
                auto it = std::find(word.begin() + i, word.end(), first);
                if (it == word.end()) {
                    new_word.insert(new_word.end(), word.begin() + i, word.end());
                    break;
                }
                new_word.insert(new_word.end(), word.begin() + i, it);
                i = static_cast<int32_t>(std::distance(word.begin(), it));

                if (word[i] == first && i < static_cast<int32_t>(word.size()) - 1 && word[i + 1] == second) {
                    new_word.push_back(first + second);
                    i += 2;
                } else {
                    new_word.push_back(word[i]);
                    i += 1;
                }
            }

            word = new_word;

            if (word.size() == 1) {
                break;
            }
            pairs = get_pairs(word);
        }

        std::u32string cond_pooled;
        for (int i = 0; i < word.size(); i++) {
            cond_pooled += word[i];
            if (i != word.size() - 1) {
                cond_pooled += utf8_to_utf32(" ");
            }
        }

        return cond_pooled;
    }

    std::vector<int> encode(std::string text)
    {
        std::string original_text = text;
        std::vector<int32_t> bpe_tokens;
        text = whitespace_clean(text);
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });

        std::regex pat(R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]|[^[:space:][:alpha:][:digit:]]+)",
            std::regex::icase);

        std::smatch matches;
        std::string str = text;
        std::vector<std::string> token_strs;
        while (std::regex_search(str, matches, pat)) {
            for (auto& token : matches) {
                std::string token_str = token.str();
                std::u32string utf32_token;
                for (int i = 0; i < token_str.length(); i++) {
                    char b = token_str[i];
                    utf32_token += byte_encoder[b];
                }
                auto bpe_strs = bpe(utf32_token);
                size_t start = 0;
                size_t pos;
                while ((pos = bpe_strs.find(' ', start)) != std::u32string::npos) {
                    auto bpe_str = bpe_strs.substr(start, pos - start);
                    bpe_tokens.push_back(encoder[bpe_str]);
                    token_strs.push_back(utf32_to_utf8(bpe_str));

                    start = pos + 1;
                }
                auto bpe_str = bpe_strs.substr(start, bpe_strs.size() - start);
                bpe_tokens.push_back(encoder[bpe_str]);
                token_strs.push_back(utf32_to_utf8(bpe_str));
            }
            str = matches.suffix();
        }
        return bpe_tokens;
    }
};

static std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text)
{
    std::vector<std::pair<std::string, float>> res;
    std::vector<int> round_brackets;
    std::vector<int> square_brackets;

    float round_bracket_multiplier = 1.1f;
    float square_bracket_multiplier = 1 / 1.1f;

    std::regex re_attention(R"(\\\(|\\\)|\\\[|\\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|\)|\]|[^\\()\[\]:]+|:)");
    std::regex re_break(R"(\s*\bBREAK\b\s*)");

    auto multiply_range = [&](int start_position, float multiplier) {
        for (int p = start_position; p < res.size(); ++p) {
            res[p].second *= multiplier;
        }
    };

    std::smatch m;
    std::string remaining_text = text;

    while (std::regex_search(remaining_text, m, re_attention)) {
        std::string text = m[0];
        std::string weight = m[1];

        if (text == "(") {
            round_brackets.push_back((int)res.size());
        } else if (text == "[") {
            square_brackets.push_back((int)res.size());
        } else if (!weight.empty()) {
            if (!round_brackets.empty()) {
                multiply_range(round_brackets.back(), std::stof(weight));
                round_brackets.pop_back();
            }
        } else if (text == ")" && !round_brackets.empty()) {
            multiply_range(round_brackets.back(), round_bracket_multiplier);
            round_brackets.pop_back();
        } else if (text == "]" && !square_brackets.empty()) {
            multiply_range(square_brackets.back(), square_bracket_multiplier);
            square_brackets.pop_back();
        } else if (text == "\\(") {
            res.push_back({ text.substr(1), 1.0f });
        } else {
            res.push_back({ text, 1.0f });
        }

        remaining_text = m.suffix();
    }

    for (int pos : round_brackets) {
        multiply_range(pos, round_bracket_multiplier);
    }

    for (int pos : square_brackets) {
        multiply_range(pos, square_bracket_multiplier);
    }

    if (res.empty()) {
        res.push_back({ "", 1.0f });
    }

    int i = 0;
    while (i + 1 < res.size()) {
        if (res[i].second == res[i + 1].second) {
            res[i].first += res[i + 1].first;
            res.erase(res.begin() + i + 1);
        } else {
            ++i;
        }
    }

    return res;
}

/*================================================ FrozenCLIPEmbedder ================================================*/
struct MultiheadAttention {
    int64_t embed_dim;
    int64_t n_head;
    bool bias;

    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;

    void create_weight_tensors(struct ggml_context* ctx)
    {
        q_proj.in_features = embed_dim;
        q_proj.out_features = embed_dim;
        q_proj.has_bias = bias;
        q_proj.create_weight_tensors(ctx);

        k_proj.in_features = embed_dim;
        k_proj.out_features = embed_dim;
        k_proj.has_bias = bias;
        k_proj.create_weight_tensors(ctx);

        v_proj.in_features = embed_dim;
        v_proj.out_features = embed_dim;
        v_proj.has_bias = bias;
        v_proj.create_weight_tensors(ctx);

        out_proj.in_features = embed_dim;
        out_proj.out_features = embed_dim;
        out_proj.has_bias = bias;
        out_proj.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char* prefix)
    {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "q_proj.");
        q_proj.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "k_proj.");
        k_proj.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "v_proj.");
        v_proj.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "out_proj.");
        out_proj.setup_weight_names(s);
    }

    // x: [N, n_token, embed_dim]
    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, bool mask = false)
    {
        int64_t N = x->ne[2];
        int64_t n_token = x->ne[1];
        int64_t d_head = embed_dim / n_head;

        struct ggml_tensor* q = q_proj.forward(ctx, x);
        q = ggml_reshape_4d(ctx, q, d_head, n_head, n_token, N); // [N, n_token, n_head, d_head]
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [N, n_head, n_token, d_head]
        q = ggml_reshape_3d(ctx, q, d_head, n_token, n_head * N); // [N * n_head, n_token, d_head]

        struct ggml_tensor* k = k_proj.forward(ctx, x);
        k = ggml_reshape_4d(ctx, k, d_head, n_head, n_token, N); // [N, n_token, n_head, d_head]
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [N, n_head, n_token, d_head]
        k = ggml_reshape_3d(ctx, k, d_head, n_token, n_head * N); // [N * n_head, n_token, d_head]

        struct ggml_tensor* v = v_proj.forward(ctx, x);
        v = ggml_reshape_4d(ctx, v, d_head, n_head, n_token, N); // [N, n_token, n_head, d_head]
        v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [N, n_head, d_head, n_token]
        v = ggml_reshape_3d(ctx, v, n_token, d_head, n_head * N); // [N * n_head, d_head, n_token]

        struct ggml_tensor* kqv = ggml_nn_attention(ctx, q, k, v, mask); // [N * n_head, n_token, d_head]

        kqv = ggml_reshape_4d(ctx, kqv, d_head, n_token, n_head, N);
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)); // [N, n_token, n_head, d_head]
        x = ggml_reshape_3d(ctx, kqv, d_head * n_head, n_token, N); // [N, n_token, d_head * n_head]

        x = out_proj.forward(ctx, x); // [N, n_token, embed_dim]
        return x;
    }
};

struct CLIPMLP {
    int64_t d_model;
    int64_t intermediate_size;
    bool use_gelu = true;

    Linear fc1;
    Linear fc2;

    void create_weight_tensors(struct ggml_context* ctx)
    {
        fc1.in_features = d_model;
        fc1.out_features = intermediate_size;
        fc1.create_weight_tensors(ctx, GGML_TYPE_F16);

        fc2.in_features = intermediate_size;
        fc2.out_features = d_model;
        fc2.create_weight_tensors(ctx, GGML_TYPE_F16);
    }

    void setup_weight_names(const char* prefix)
    {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "fc1.");
        fc1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "fc2.");
        fc2.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x)
    {
        // x: [N, n_token, d_model]

        x = fc1.forward(ctx, x);
        if (use_gelu) {
            x = ggml_gelu_inplace(ctx, x);
        } else {
            x = ggml_gelu_quick_inplace(ctx, x);
        }
        x = fc2.forward(ctx, x);
        return x;
    }
};

struct CLIPLayer {
    int64_t d_model; // hidden_size/embed_dim
    int64_t n_head;
    int64_t intermediate_size;

    MultiheadAttention self_attn;
    LayerNorm layer_norm1;
    LayerNorm layer_norm2;
    CLIPMLP mlp;

    void create_weight_tensors(struct ggml_context* ctx)
    {
        self_attn.embed_dim = d_model;
        self_attn.n_head = n_head;
        self_attn.bias = true;
        self_attn.create_weight_tensors(ctx);

        layer_norm1.normalized_shape = d_model;
        layer_norm1.create_weight_tensors(ctx);
        layer_norm2.normalized_shape = d_model;
        layer_norm2.create_weight_tensors(ctx);

        mlp.d_model = d_model;
        mlp.intermediate_size = intermediate_size;
        // mlp.use_gelu = true;
        mlp.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char* prefix)
    {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "self_attn.");
        self_attn.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer_norm1.");
        layer_norm1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "layer_norm2.");
        layer_norm2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "mlp.");
        mlp.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, bool mask = true)
    {
        // x: [N, n_token, d_model]
        // f32 [   768,    77,     1,     1], x
        x = ggml_add(ctx, x, self_attn.forward(ctx, layer_norm1.forward(ctx, x), mask));
        x = ggml_add(ctx, x, mlp.forward(ctx, layer_norm2.forward(ctx, x)));

        return x; // f32 [   768,    77,     1,     1], x 
    }
};

struct CLIPEncoder {
    int64_t n_layer = 32; // max -- 32 ?

    int64_t d_model;
    int64_t n_head;
    int64_t intermediate_size;

    CLIPLayer layers[32];

    void create_weight_tensors(struct ggml_context* ctx)
    {
        for (int i = 0; i < n_layer; i++) {
            layers[i].d_model = d_model;
            layers[i].n_head = n_head;
            layers[i].intermediate_size = intermediate_size;
            layers[i].create_weight_tensors(ctx);
        }
    }

    void setup_weight_names(const char* prefix)
    {
        char s[GGML_MAX_NAME];

        for (int i = 0; i < n_layer; i++) {
            snprintf(s, sizeof(s), "%s%s%d.", prefix, "layers.", i);
            layers[i].setup_weight_names(s);
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, int clip_skip = -1, bool mask = true)
    {
        // x: [N, n_token, d_model]
        int layer_idx = n_layer - 1;
        if (clip_skip > 0) {
            layer_idx = n_layer - clip_skip;
        }

        for (int i = 0; i < n_layer; i++) {
            if (i == layer_idx + 1) {
                break;
            }
            x = layers[i].forward(ctx, x, mask); // [N, n_token, d_model]
        }
        return x;
    }
};

struct CLIPEmbeddings {
    int64_t embed_dim; // 768, 1024, 1280 ...
    int64_t vocab_size = 49408;
    int64_t num_positions = 77;

    struct ggml_tensor* token_embedding_weight;
    struct ggml_tensor* position_embedding_weight;

    void create_weight_tensors(struct ggml_context* ctx)
    {
        token_embedding_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, embed_dim, vocab_size); // [768, 49408, 1, 1]
        position_embedding_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, num_positions); // [768, 77, 1, 1]
    }

    void setup_weight_names(const char* prefix)
    {
        ggml_format_name(token_embedding_weight, "%s%s", prefix, "token_embedding.weight");
        ggml_format_name(position_embedding_weight, "%s%s", prefix, "position_embedding.weight");
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* input_ids)
    {
        // input_ids //    i32 [    77,     1,     1,     1], net.input_0
        GGML_ASSERT(input_ids->ne[0] == position_embedding_weight->ne[1]);
        input_ids = ggml_reshape_3d(ctx, input_ids, input_ids->ne[0], 1, input_ids->ne[1]);
        // input_ids = ggml_cast(ctx, input_ids, GGML_TYPE_I32); // force f32 to i32
        auto token_embedding = ggml_get_rows(ctx, token_embedding_weight, input_ids);
        token_embedding = ggml_reshape_3d(ctx, token_embedding, token_embedding->ne[0], token_embedding->ne[1], token_embedding->ne[3]);

        // f32 [   768,    77,     1,     1],  token_embedding
        // f32 [   768,    77,     1,     1],  position_embedding.weight
        auto x = ggml_add(ctx, token_embedding, position_embedding_weight); // [N, n_token, embed_dim]
        return x; // f32 [   768,    77,     1,     1], x
    }
};

enum CLIPVersion {
    OPENAI_CLIP_VIT_L_14, // SD 1.x and SDXL
    OPEN_CLIP_VIT_H_14, // SD 2.x
    OPEN_CLIP_VIT_BIGG_14, // SDXL
};

struct CLIPTextModel {
    CLIPVersion version = OPENAI_CLIP_VIT_L_14;
    int32_t vocab_size = 49408;
    int32_t n_token = 77; // max_position_embeddings
    int32_t hidden_size = 768;
    int32_t intermediate_size = 3072;
    int32_t n_head = 12;
    int32_t n_layer = 12; // num_hidden_layers
    int32_t projection_dim = 1280; // only for OPEN_CLIP_VIT_BIGG_14
    int32_t clip_skip = 2;

    struct ggml_tensor* text_projection;
    CLIPEmbeddings embeddings;
    CLIPEncoder encoder;
    LayerNorm final_layer_norm;

    // ----------------------------------------------------------------------------------
    void create_weight_tensors(struct ggml_context* ctx)
    {
        if (version == OPEN_CLIP_VIT_H_14) {
            hidden_size = 1024;
            intermediate_size = 4096;
            n_head = 16;
            n_layer = 24;
        } else if (version == OPEN_CLIP_VIT_BIGG_14) { // CLIPTextModelWithProjection
            hidden_size = 1280;
            intermediate_size = 5120;
            n_head = 20;
            n_layer = 32;
        }

        if (version == OPEN_CLIP_VIT_BIGG_14) {
            text_projection = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, projection_dim, hidden_size); // OPEN_CLIP_VIT_BIGG_14
        }

        embeddings.embed_dim = hidden_size; // 1024, 1280 ...
        embeddings.vocab_size = vocab_size;
        embeddings.num_positions = n_token;
        embeddings.create_weight_tensors(ctx);

        encoder.n_layer = n_layer;
        encoder.d_model = hidden_size;
        encoder.n_head = n_head;
        encoder.intermediate_size = intermediate_size;
        encoder.create_weight_tensors(ctx);

        final_layer_norm.normalized_shape = hidden_size;
        final_layer_norm.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char* prefix)
    {
        char s[GGML_MAX_NAME];

        if (version == OPEN_CLIP_VIT_BIGG_14) {
            ggml_format_name(text_projection, "%s%s", prefix, "text_projection");
        }

        snprintf(s, sizeof(s), "%s%s", prefix, "embeddings.");
        embeddings.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "encoder.");
        encoder.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "final_layer_norm.");
        final_layer_norm.setup_weight_names(s);
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* input_ids,
        size_t max_token_idx = 0, bool return_pooled = false)
    {
        auto x = embeddings.forward(ctx, input_ids); // [N, n_token, hidden_size], xxxx_debug
        x = encoder.forward(ctx, x, return_pooled ? -1 : clip_skip, true);
        if (return_pooled) {
            x = final_layer_norm.forward(ctx, x);
        }

        if (return_pooled) {
            // auto text_projection = params["text_projection"];
            ggml_tensor* pooled = ggml_view_1d(ctx, x, hidden_size, x->nb[1] * max_token_idx);
            pooled = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, text_projection)), pooled);
            return pooled;
        }

        return x; // [N, n_token, hidden_size]
    }
};

struct TextEncoder : GGMLNetwork {
    size_t max_token_idx = 0;
    bool return_pooled = false;

    CLIPTokenizer tokenizer;
    CLIPTextModel text_model;
    CLIPTextModel text_model2;

    void create_weight_tensors(struct ggml_context* ctx)
    {
        // clip_skip = 2;
        text_model.version = OPENAI_CLIP_VIT_L_14;
        text_model.clip_skip = 2;
        text_model.create_weight_tensors(ctx);

        text_model2.version = OPEN_CLIP_VIT_BIGG_14;
        text_model2.clip_skip = 2;
        text_model2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char* prefix)
    {
        char s[GGML_MAX_NAME];

        // snprintf(s, sizeof(s), "%s%s", prefix, "transformer.text_model.");
        snprintf(s, sizeof(s), "%s%s", prefix, "text_model.");
        text_model.setup_weight_names(s);

        // snprintf(s, sizeof(s), "%s%s", prefix, "1.transformer.text_model.");
        snprintf(s, sizeof(s), "%s%s", prefix, "text_model2.");
        text_model2.setup_weight_names(s);
    }

    // size_t get_graph_size()
    // {
    //     return GGML_DEFAULT_GRAPH_SIZE * 2; // 2048 * 2
    // }

    // inputs must be GGML_TYPE_I32, override !!! 
    void create_input_tensors(int argc, TENSOR *argv[], struct ggml_context *ctx, struct ggml_tensor *x[])
    {
        char input_name[64];
        for (int i = 0; i < argc; i++) {
            if (argv[i] == NULL) {
                x[i] = NULL;
                continue;
            }

            snprintf(input_name, sizeof(input_name), "net.input_%d", i);
            x[i] = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 
                (int64_t)argv[i]->width, (int64_t)argv[i]->height, (int64_t)argv[i]->chan, (int64_t)argv[i]->batch);
            ggml_set_name(x[i], input_name);
        }
    }

    // inputs must be GGML_TYPE_I32, override !!! 
    void setup_input_values(int argc, TENSOR *argv[], bool is_cpu_backend, struct ggml_tensor *x[])
    {
        std::vector<int> temp;

        if (is_cpu_backend) {
            syslog_debug("Set input values to cpu backend ...");
        } else {
            syslog_debug("Set input values to cuda backend ...");
        }

        for (int i = 0; i < argc; i++) {
            if (argv[i] == NULL)
                continue;

            int n = x[i]->ne[0] * x[i]->ne[1] * x[i]->ne[2] * x[i]->ne[3];
            temp.reserve(n);
            for (int j = 0; j < n; j++) {
                temp[j] = (int)argv[i]->data[j]; // force float to int32
            }

            if (is_cpu_backend) {
                memcpy(x[i]->data, temp.data(), ggml_nbytes(x[i]));
            } else {
                ggml_backend_tensor_set(x[i], temp.data(), 0, ggml_nbytes(x[i]));
            }
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, int argc, struct ggml_tensor* argv[2])
    {
        struct ggml_tensor* input_ids = argv[0];  // i32 [    77,     1,     1,     1], net.input_0
        struct ggml_tensor* input_ids2 = argv[1]; // i32 [    77,     1,     1,     1], net.input_1

        size_t N = input_ids->ne[1];
        size_t n_token = input_ids->ne[0];

        if (input_ids != NULL && input_ids->ne[0] > text_model.n_token) {
            GGML_ASSERT(input_ids->ne[0] % text_model.n_token == 0);
            input_ids = ggml_reshape_2d(ctx, input_ids, text_model.n_token, input_ids->ne[0] / text_model.n_token);
        }
        if (input_ids2 != NULL && input_ids2->ne[0] > text_model2.n_token) {
            GGML_ASSERT(input_ids2->ne[0] % text_model2.n_token == 0);
            input_ids2 = ggml_reshape_2d(ctx, input_ids2, text_model2.n_token, input_ids2->ne[0] / text_model2.n_token);
        }

        if (return_pooled) {
            return text_model2.forward(ctx, input_ids2, max_token_idx, return_pooled);
        }

        auto hidden_states = text_model.forward(ctx, input_ids, max_token_idx, return_pooled); // [N, n_token, hidden_size]
        hidden_states = ggml_reshape_4d(ctx,
            hidden_states,
            hidden_states->ne[0],
            hidden_states->ne[1],
            hidden_states->ne[2],
            hidden_states->ne[3]);
        hidden_states = ggml_cont(ctx, ggml_permute(ctx, hidden_states, 2, 0, 1, 3));

        auto hidden_states2 = text_model2.forward(ctx, input_ids2, max_token_idx, return_pooled); //  f32 [  1280,    77,     1,     1]

        hidden_states2 = ggml_reshape_4d(ctx,
            hidden_states2,
            hidden_states2->ne[0],
            hidden_states2->ne[1],
            hidden_states2->ne[2],
            hidden_states2->ne[3]);

        hidden_states2 = ggml_cont(ctx, ggml_permute(ctx, hidden_states2, 2, 0, 1, 3));
        hidden_states = ggml_concat(ctx, hidden_states, hidden_states2, 2); // [N, n_token, hidden_size + hidden_size2]

        hidden_states = ggml_cont(ctx, ggml_permute(ctx, hidden_states, 1, 2, 0, 3));
        hidden_states = ggml_reshape_3d(ctx, hidden_states, hidden_states->ne[0], n_token, N);

        return hidden_states; // f32 [  2048,    77,     1,     1],  (permuted) (cont) (reshaped)
    }

    void pad_tokens(std::vector<int>& tokens, std::vector<float>& weights, bool padding = false)
    {
        if (!padding)
            return;

        size_t max_length = text_model.n_token; // text_model.n_token = 77
        size_t n = std::ceil(tokens.size() * 1.0 / (max_length - 2));
        if (n == 0) {
            n = 1;
        }
        size_t length = max_length * n;

        std::vector<int> new_tokens;
        std::vector<float> new_weights;
        new_tokens.push_back(BOS_TOKEN_ID);
        new_weights.push_back(1.0);
        int token_idx = 0;
        for (int i = 1; i < length; i++) {
            if (token_idx >= tokens.size()) {
                break;
            }
            if (i % max_length == 0) {
                new_tokens.push_back(BOS_TOKEN_ID);
                new_weights.push_back(1.0);
            } else if (i % max_length == max_length - 1) {
                new_tokens.push_back(EOS_TOKEN_ID);
                new_weights.push_back(1.0);
            } else {
                new_tokens.push_back(tokens[token_idx]);
                new_weights.push_back(weights[token_idx]);
                token_idx++;
            }
        }

        new_tokens.push_back(EOS_TOKEN_ID);
        new_weights.push_back(1.0);
        tokens = new_tokens;
        weights = new_weights;

        // if (padding)
        {
            tokens.insert(tokens.end(), length - tokens.size(), PAD_TOKEN_ID);
            weights.insert(weights.end(), length - weights.size(), 1.0);
        }
    }

    std::pair<std::vector<int>, std::vector<float>> tokenize(std::string text, bool padding = false)
    {
        std::vector<int> tokens;
        std::vector<float> weights;

        // size_t max_length = text_model.n_token;
        auto parsed_attention = parse_prompt_attention(text);
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight = item.second;
            std::vector<int> curr_tokens = tokenizer.encode(curr_text);

            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }

        // pad_tokens(tokens, weights, max_length, padding);
        pad_tokens(tokens, weights, padding);

        return { tokens, weights };
    }
};

std::vector<TENSOR *> clip_encode(TextEncoder *clip, char *text, int height, int width);

#endif // __CLIP_H__
