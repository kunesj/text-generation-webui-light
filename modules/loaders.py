import functools
from collections import OrderedDict

import gradio as gr

loaders_and_params = OrderedDict(
    {
        "llama.cpp": [
            "gpu_layers",
            "cpu_moe",
            "threads",
            "threads_batch",
            "batch_size",
            "ubatch_size",
            "ctx_size",
            "cache_type",
            "tensor_split",
            "extra_flags",
            "streaming_llm",
            "rope_freq_base",
            "compress_pos_emb",
            "row_split",
            "no_kv_offload",
            "no_mmap",
            "mlock",
            "numa",
            "model_draft",
            "draft_max",
            "gpu_layers_draft",
            "device_draft",
            "ctx_size_draft",
            "speculative_decoding_accordion",
            "mmproj",
            "mmproj_accordion",
            "vram_info",
        ],
    }
)

loaders_samplers = {
    "llama.cpp": {
        "temperature",
        "dynatemp_low",
        "dynatemp_high",
        "dynatemp_exponent",
        "min_p",
        "top_p",
        "top_k",
        "typical_p",
        "xtc_threshold",
        "xtc_probability",
        "top_n_sigma",
        "dry_multiplier",
        "dry_allowed_length",
        "dry_base",
        "repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty_range",
        "mirostat_mode",
        "mirostat_tau",
        "mirostat_eta",
        "dynamic_temperature",
        "temperature_last",
        "auto_max_new_tokens",
        "ban_eos_token",
        "add_bos_token",
        "enable_thinking",
        "reasoning_effort",
        "seed",
        "sampler_priority",
        "dry_sequence_breakers",
        "grammar_string",
        "grammar_file_row",
    },
}


@functools.cache
def list_all_samplers():
    all_samplers = set()
    for k in loaders_samplers:
        for sampler in loaders_samplers[k]:
            all_samplers.add(sampler)

    return sorted(all_samplers)


def blacklist_samplers(loader, dynamic_temperature):
    all_samplers = list_all_samplers()
    output = []

    for sampler in all_samplers:
        if loader == "All" or sampler in loaders_samplers[loader]:
            if sampler.startswith("dynatemp"):
                output.append(gr.update(visible=dynamic_temperature))
            else:
                output.append(gr.update(visible=True))
        else:
            output.append(gr.update(visible=False))

    return output


@functools.cache
def get_all_params():
    all_params = set()
    for k in loaders_and_params:
        for el in loaders_and_params[k]:
            all_params.add(el)

    return sorted(all_params)


def make_loader_params_visible(loader):
    params = []
    all_params = get_all_params()
    if loader in loaders_and_params:
        params = loaders_and_params[loader]

    return [gr.update(visible=True) if k in params else gr.update(visible=False) for k in all_params]
