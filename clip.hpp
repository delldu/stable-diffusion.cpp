#ifndef __CLIP_HPP__
#define __CLIP_HPP__

#include "ggml_extend.hpp"
#include "model.h"

/*================================================== CLIPTokenizer ===================================================*/

std::pair<std::unordered_map<std::string, float>, std::string> extract_and_remove_lora(std::string text) {
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

const std::string UNK_TOKEN = "<|endoftext|>";
const std::string BOS_TOKEN = "<|startoftext|>";
const std::string EOS_TOKEN = "<|endoftext|>";
const std::string PAD_TOEKN = "<|endoftext|>";

const int UNK_TOKEN_ID = 49407;
const int BOS_TOKEN_ID = 49406;
const int EOS_TOKEN_ID = 49407;
const int PAD_TOKEN_ID = 49407;

std::vector<std::pair<int, std::u32string>> bytes_to_unicode() {
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
    // LOG_DEBUG("byte_unicode_pairs %d", byte_unicode_pairs.size());
    return byte_unicode_pairs;
}

// Ref: https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py

typedef std::function<bool(std::string&, std::vector<int32_t>&)> on_new_token_cb_t;

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

    static std::string strip(const std::string& str) {
        std::string::size_type start = str.find_first_not_of(" \t\n\r\v\f");
        std::string::size_type end   = str.find_last_not_of(" \t\n\r\v\f");

        if (start == std::string::npos) {
            // String contains only whitespace characters
            return "";
        }

        return str.substr(start, end - start + 1);
    }

    static std::string whitespace_clean(std::string text) {
        text = std::regex_replace(text, std::regex(R"(\s+)"), " ");
        text = strip(text);
        return text;
    }

    static std::set<std::pair<std::u32string, std::u32string>> get_pairs(const std::vector<std::u32string>& subwords) {
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
        : version(version) {}

    // xxxx_debug
    void load_from_merges(const std::string& merges_utf8_str) {
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
        // LOG_DEBUG("merges size %llu", merges.size());
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
        LOG_DEBUG("vocab size: %llu", vocab.size());
        int i = 0;
        for (const auto& token : vocab) {
            encoder[token] = i;
            decoder[i]     = token;
            i++;
        }
        encoder_len = i;

        auto it = encoder.find(utf8_to_utf32("img</w>"));
        if (it != encoder.end()) {
            LOG_DEBUG(" trigger word img already in vocab");
        } else {
            LOG_DEBUG(" trigger word img not in vocab yet");
        }

        int rank = 0;
        for (const auto& merge : merge_pairs) {
            bpe_ranks[merge] = rank++;
        }
        bpe_len = rank;
    };

    std::u32string bpe(const std::u32string& token) {
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

            std::u32string first  = bigram.first;
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

        std::u32string result;
        for (int i = 0; i < word.size(); i++) {
            result += word[i];
            if (i != word.size() - 1) {
                result += utf8_to_utf32(" ");
            }
        }

        return result;
    }

    // std::vector<int> encode(std::string text, on_new_token_cb_t on_new_token_cb) {
    std::vector<int> encode(std::string text) {
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
                size_t start  = 0;
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


std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text) {
    std::vector<std::pair<std::string, float>> res;
    std::vector<int> round_brackets;
    std::vector<int> square_brackets;

    float round_bracket_multiplier  = 1.1f;
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
        std::string text   = m[0];
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
            res.push_back({text.substr(1), 1.0f});
        } else {
            res.push_back({text, 1.0f});
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
        res.push_back({"", 1.0f});
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

// Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py

struct CLIPMLP : public GGMLBlock {
protected:
    bool use_gelu;

public:
    CLIPMLP(int64_t d_model, int64_t intermediate_size) {
        blocks["fc1"] = std::shared_ptr<GGMLBlock>(new Linear(d_model, intermediate_size));
        blocks["fc2"] = std::shared_ptr<GGMLBlock>(new Linear(intermediate_size, d_model));

        if (d_model == 1024 || d_model == 1280) {  // SD 2.x
            use_gelu = true;
        } else {  // SD 1.x
            use_gelu = false;
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
        // x: [N, n_token, d_model]
        auto fc1 = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
        auto fc2 = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);

        x = fc1->forward(ctx, x);
        if (use_gelu) {
            x = ggml_gelu_inplace(ctx, x);
        } else {
            x = ggml_gelu_quick_inplace(ctx, x);
        }
        x = fc2->forward(ctx, x);
        return x;
    }
};

struct CLIPLayer : public GGMLBlock {
protected:
    int64_t d_model;  // hidden_size/embed_dim
    int64_t n_head;
    int64_t intermediate_size;

public:
    CLIPLayer(int64_t d_model,
              int64_t n_head,
              int64_t intermediate_size)
        : d_model(d_model),
          n_head(n_head),
          intermediate_size(intermediate_size) {
        blocks["self_attn"] = std::shared_ptr<GGMLBlock>(new MultiheadAttention(d_model, n_head, true));

        blocks["layer_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(d_model));
        blocks["layer_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(d_model));

        blocks["mlp"] = std::shared_ptr<GGMLBlock>(new CLIPMLP(d_model, intermediate_size));
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, bool mask = true) {
        // x: [N, n_token, d_model]
        auto self_attn   = std::dynamic_pointer_cast<MultiheadAttention>(blocks["self_attn"]);
        auto layer_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm1"]);
        auto layer_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm2"]);
        auto mlp         = std::dynamic_pointer_cast<CLIPMLP>(blocks["mlp"]);

        x = ggml_add(ctx, x, self_attn->forward(ctx, layer_norm1->forward(ctx, x), mask));
        x = ggml_add(ctx, x, mlp->forward(ctx, layer_norm2->forward(ctx, x)));
        return x;
    }
};

struct CLIPEncoder : public GGMLBlock {
protected:
    int64_t n_layer;

public:
    CLIPEncoder(int64_t n_layer,
                int64_t d_model,
                int64_t n_head,
                int64_t intermediate_size)
        : n_layer(n_layer) {
        for (int i = 0; i < n_layer; i++) {
            std::string name = "layers." + std::to_string(i);
            blocks[name]     = std::shared_ptr<GGMLBlock>(new CLIPLayer(d_model, n_head, intermediate_size));
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, int clip_skip = -1, bool mask = true) {
        // x: [N, n_token, d_model]
        int layer_idx = n_layer - 1;
        if (clip_skip > 0) {
            layer_idx = n_layer - clip_skip;
        }

        for (int i = 0; i < n_layer; i++) {
            // LOG_DEBUG("layer %d", i);
            if (i == layer_idx + 1) {
                break;
            }
            std::string name = "layers." + std::to_string(i);
            auto layer       = std::dynamic_pointer_cast<CLIPLayer>(blocks[name]);
            x                = layer->forward(ctx, x, mask);  // [N, n_token, d_model]
            // LOG_DEBUG("layer %d", i);
        }
        return x;
    }
};

class CLIPEmbeddings : public GGMLBlock {
protected:
    int64_t embed_dim; // 1024, 1280 ...
    int64_t vocab_size;
    int64_t num_positions;

    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        params["token_embedding.weight"]    = ggml_new_tensor_2d(ctx, wtype, embed_dim, vocab_size);
        params["position_embedding.weight"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_dim, num_positions);
    }

public:
    CLIPEmbeddings(int64_t embed_dim,
                   int64_t vocab_size    = 49408,
                   int64_t num_positions = 77)
        : embed_dim(embed_dim),
          vocab_size(vocab_size),
          num_positions(num_positions) {
    }


    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* input_ids) {
        // input_ids: [N, n_token]
        auto token_embed_weight    = params["token_embedding.weight"];
        auto position_embed_weight = params["position_embedding.weight"];

        GGML_ASSERT(input_ids->ne[0] == position_embed_weight->ne[1]);
        input_ids            = ggml_reshape_3d(ctx, input_ids, input_ids->ne[0], 1, input_ids->ne[1]);
        auto token_embedding = ggml_get_rows(ctx, token_embed_weight, input_ids);
        token_embedding      = ggml_reshape_3d(ctx, token_embedding, 
            token_embedding->ne[0], token_embedding->ne[1], token_embedding->ne[3]);

        // token_embedding + position_embedding
        auto x = ggml_add(ctx,
                          token_embedding,
                          position_embed_weight);  // [N, n_token, embed_dim]
        return x;
    }
};


enum CLIPVersion {
    OPENAI_CLIP_VIT_L_14,   // SD 1.x and SDXL
    OPEN_CLIP_VIT_H_14,     // SD 2.x
    OPEN_CLIP_VIT_BIGG_14,  // SDXL
};

class CLIPTextModel : public GGMLBlock {
protected:
    void init_params(struct ggml_context* ctx, ggml_type wtype) {
        if (version == OPEN_CLIP_VIT_BIGG_14) {
            // CheckPoint("OPEN_CLIP_VIT_BIGG_14 projection_dim = %d, hidden_size = %d", projection_dim, hidden_size);
            // OPEN_CLIP_VIT_BIGG_14 projection_dim = 1280, hidden_size = 1280
            params["text_projection"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, projection_dim, hidden_size);
        }
    }

public:
    CLIPVersion version = OPENAI_CLIP_VIT_L_14;
    // network hparams
    int32_t vocab_size        = 49408;
    int32_t n_token           = 77;  // max_position_embeddings
    int32_t hidden_size       = 768;
    int32_t intermediate_size = 3072;
    int32_t n_head            = 12;
    int32_t n_layer           = 12;    // num_hidden_layers
    int32_t projection_dim    = 1280;  // only for OPEN_CLIP_VIT_BIGG_14
    int32_t clip_skip         = -1;
    bool with_final_ln        = true;

    CLIPTextModel(CLIPVersion version = OPENAI_CLIP_VIT_L_14,
                  int clip_skip_value = -1,
                  bool with_final_ln  = true)
        : version(version), with_final_ln(with_final_ln) {
        if (version == OPEN_CLIP_VIT_H_14) {
            hidden_size       = 1024;
            intermediate_size = 4096;
            n_head            = 16;
            n_layer           = 24;
        } else if (version == OPEN_CLIP_VIT_BIGG_14) {  // CLIPTextModelWithProjection
            hidden_size       = 1280;
            intermediate_size = 5120;
            n_head            = 20;
            n_layer           = 32;
        }

        // CheckPoint("CLIPVersion version = %d, clip_skip_value = %d", version, clip_skip_value);
        // **** CLIPVersion version = 0, clip_skip_value = -1
        // CLIPVersion version = 0, clip_skip_value = 2 --- OPENAI_CLIP_VIT_L_14
        // CLIPVersion version = 2, clip_skip_value = 2 --- OPEN_CLIP_VIT_BIGG_14

        set_clip_skip(clip_skip_value);


        blocks["embeddings"]       = std::shared_ptr<GGMLBlock>(new CLIPEmbeddings(hidden_size, vocab_size, n_token));
        blocks["encoder"]          = std::shared_ptr<GGMLBlock>(new CLIPEncoder(n_layer, hidden_size, n_head, intermediate_size));
        blocks["final_layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(hidden_size));
    }

    // xxxx_debug
    void set_clip_skip(int skip) {
        if (skip <= 0) {
            return;
        }
        clip_skip = skip;
    }


    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* input_ids,
                                size_t max_token_idx = 0,
                                bool return_pooled   = false) {
        // CheckPoint("max_token_idx = %ld", max_token_idx); // 0, 1, 4, 17, 21 ...

        // input_ids: [N, n_token]
        auto embeddings       = std::dynamic_pointer_cast<CLIPEmbeddings>(blocks["embeddings"]);
        auto encoder          = std::dynamic_pointer_cast<CLIPEncoder>(blocks["encoder"]);
        auto final_layer_norm = std::dynamic_pointer_cast<LayerNorm>(blocks["final_layer_norm"]);

        auto x = embeddings->forward(ctx, input_ids);  // [N, n_token, hidden_size], xxxx_debug
        x      = encoder->forward(ctx, x, return_pooled ? -1 : clip_skip, true);
        if (return_pooled || with_final_ln) {
            x = final_layer_norm->forward(ctx, x);
        }

        if (return_pooled) {
            auto text_projection = params["text_projection"];
            ggml_tensor* pooled  = ggml_view_1d(ctx, x, hidden_size, x->nb[1] * max_token_idx);
            pooled               = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, text_projection)), pooled);
            return pooled;
        }

        return x;  // [N, n_token, hidden_size]
    }
};

// ldm.modules.encoders.modules.FrozenCLIPEmbedder
// Ref: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/cad87bf4e3e0b0a759afa94e933527c3123d59bc/modules/sd_hijack_clip.py#L283
struct FrozenCLIPEmbedderWithCustomWords : public GGMLModule {
    SDVersion version = VERSION_1_x;
    CLIPTokenizer tokenizer;
    CLIPTextModel text_model;
    CLIPTextModel text_model2;

    // std::vector<uint8_t> token_embed_custom;
    // std::vector<std::string> readed_embeddings;

    FrozenCLIPEmbedderWithCustomWords(ggml_backend_t backend,
                                      ggml_type wtype,
                                      SDVersion version = VERSION_1_x,
                                      int clip_skip     = -1)
        : GGMLModule(backend, wtype), version(version), tokenizer(version) {

        // CheckPoint("FrozenCLIPEmbedderWithCustomWords: version = %d, clip_skip = %d", version, clip_skip);
        // FrozenCLIPEmbedderWithCustomWords: version = 2, clip_skip = -1
        if (clip_skip <= 0) {
            clip_skip = 1;
            if (version == VERSION_2_x || version == VERSION_XL) {
                clip_skip = 2;
            }
        }
        if (version == VERSION_1_x) {
            text_model = CLIPTextModel(OPENAI_CLIP_VIT_L_14, clip_skip, true /*with_final_ln*/);
            text_model.init(params_ctx, wtype);
        } else if (version == VERSION_2_x) {
            text_model = CLIPTextModel(OPEN_CLIP_VIT_H_14, clip_skip, true /*with_final_ln*/);
            text_model.init(params_ctx, wtype);
        } else if (version == VERSION_XL) {
            text_model  = CLIPTextModel(OPENAI_CLIP_VIT_L_14, clip_skip, false /*with_final_ln*/);
            text_model2 = CLIPTextModel(OPEN_CLIP_VIT_BIGG_14, clip_skip, false /*with_final_ln*/);
            text_model.init(params_ctx, wtype);
            text_model2.init(params_ctx, wtype);
        }
    }

    std::string get_desc() {
        return "clip";
    }

    void set_clip_skip(int clip_skip) {
        text_model.set_clip_skip(clip_skip);
        if (version == VERSION_XL) {
            text_model2.set_clip_skip(clip_skip);
        }
    }

    // std::map<std::string, struct ggml_tensor*>& tensors !!!
    void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
        text_model.get_param_tensors(tensors, prefix + "transformer.text_model");
        if (version == VERSION_XL) {
            // cond_stage_model.1.transformer.text_model.encoder.layers.9.self_attn.v_proj.weight
            text_model2.get_param_tensors(tensors, prefix + "1.transformer.text_model");
        }
    }

    struct ggml_tensor* forward(struct ggml_context* ctx,
                                struct ggml_tensor* input_ids,
                                struct ggml_tensor* input_ids2, // negative prompt ...
                                size_t max_token_idx = 0,
                                bool return_pooled   = false) {
        // CheckPoint("return_pooled = %d", return_pooled); // false | true
        size_t N       = input_ids->ne[1];
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

        auto hidden_states = text_model.forward(ctx, input_ids);  // [N, n_token, hidden_size]
        if (version == VERSION_XL) {
            hidden_states = ggml_reshape_4d(ctx,
                                            hidden_states,
                                            hidden_states->ne[0],
                                            hidden_states->ne[1],
                                            hidden_states->ne[2],
                                            hidden_states->ne[3]);
            hidden_states = ggml_cont(ctx, ggml_permute(ctx, hidden_states, 2, 0, 1, 3));

            auto hidden_states2 = text_model2.forward(ctx, input_ids2);  // [N, n_token, hidden_size2]
            hidden_states2 = ggml_reshape_4d(ctx,
                                             hidden_states2,
                                             hidden_states2->ne[0],
                                             hidden_states2->ne[1],
                                             hidden_states2->ne[2],
                                             hidden_states2->ne[3]);
            hidden_states2 = ggml_cont(ctx, ggml_permute(ctx, hidden_states2, 2, 0, 1, 3));

            hidden_states = ggml_concat(ctx, hidden_states, hidden_states2, 2);  // [N, n_token, hidden_size + hidden_size2]

            hidden_states = ggml_cont(ctx, ggml_permute(ctx, hidden_states, 1, 2, 0, 3));
        }
        hidden_states = ggml_reshape_3d(ctx, hidden_states, hidden_states->ne[0], n_token, N);
        return hidden_states;
    }

    struct ggml_cgraph* build_graph(struct ggml_tensor* input_ids,
                                    struct ggml_tensor* input_ids2 = NULL,
                                    size_t max_token_idx           = 0,
                                    bool return_pooled             = false) {
        struct ggml_cgraph* gf = ggml_new_graph(compute_ctx);

        input_ids2 = to_backend(input_ids2);
        if (!return_pooled) {
            input_ids = to_backend(input_ids);
        }

        struct ggml_tensor* hidden_states = forward(compute_ctx, input_ids, input_ids2, max_token_idx, return_pooled);

        ggml_build_forward_expand(gf, hidden_states);

        return gf;
    }

    // xxxx_debug
    void compute(const int n_threads,
                 struct ggml_tensor* input_ids,
                 struct ggml_tensor* input_ids2,
                 size_t max_token_idx,
                 bool return_pooled,
                 ggml_tensor** output,
                 ggml_context* output_ctx = NULL) {
        auto get_graph = [&]() -> struct ggml_cgraph* {
            return build_graph(input_ids, input_ids2, max_token_idx, return_pooled);
        };
        GGMLModule::compute(get_graph, n_threads, true, output, output_ctx);
    }

    // xxxx_debug
    std::pair<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                             bool padding = false) {
        return tokenize(text, text_model.n_token, padding);
    }


    void pad_tokens(std::vector<int>& tokens,
                    std::vector<float>& weights,
                    size_t max_length = 0,
                    bool padding      = false) {
        if (max_length > 0 && padding) {
            size_t n = std::ceil(tokens.size() * 1.0 / (max_length - 2));
            if (n == 0) {
                n = 1;
            }
            size_t length = max_length * n;
            LOG_DEBUG("token length: %llu", length);
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
            tokens  = new_tokens;
            weights = new_weights;

            if (padding) {
                int pad_token_id = PAD_TOKEN_ID;
                if (version == VERSION_2_x) {
                    pad_token_id = 0;
                }
                tokens.insert(tokens.end(), length - tokens.size(), pad_token_id);
                weights.insert(weights.end(), length - weights.size(), 1.0);
            }
        }
    }


    std::pair<std::vector<int>, std::vector<float>> tokenize(std::string text,
                                                             size_t max_length = 0,
                                                             bool padding      = false) {
        auto parsed_attention = parse_prompt_attention(text);
        std::vector<int> tokens;
        std::vector<float> weights;
        for (const auto& item : parsed_attention) {
            const std::string& curr_text = item.first;
            float curr_weight            = item.second;
            std::vector<int> curr_tokens = tokenizer.encode(curr_text);

            tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
            weights.insert(weights.end(), curr_tokens.size(), curr_weight);
        }

        pad_tokens(tokens, weights, max_length, padding);

        return {tokens, weights};
    }
};


#endif  // __CLIP_HPP__
