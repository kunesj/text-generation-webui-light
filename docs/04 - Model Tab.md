This is where you load models, apply LoRAs to a loaded model, and download new models.

## Model loaders

### llama.cpp

Loads: GGUF models. Note: GGML models have been deprecated and do not work anymore.

Example: https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF

* **n-gpu-layers**: The number of layers to allocate to the GPU. If set to 0, only the CPU will be used. If you want to offload all layers, you can simply set this to the maximum value.
* **n_ctx**: Context length of the model. In llama.cpp, the cache is preallocated, so the higher this value, the higher the VRAM. It is automatically set to the maximum sequence length for the model based on the metadata inside the GGUF file, but you may need to lower this value be able to fit the model into your GPU. After loading the model, the "Truncate the prompt up to this length" parameter under "Parameters" > "Generation" is automatically set to your chosen "n_ctx" so that you don't have to set the same thing twice.
* **tensor_split**: For multi-gpu only. Sets the amount of memory to allocate per GPU as proportions. Not to be confused with other loaders where this is set in GB; here you can set something like `30,70` for 30%/70%.
* **n_batch**: Batch size for prompt processing. Higher values are supposed to make generation faster, but I have never obtained any benefit from changing this value.
* **threads**: Number of threads. Recommended value: your number of physical cores. 
* **threads_batch**: Number of threads for batch processing. Recommended value: your total number of cores (physical + virtual).
* **tensorcores**: Use llama.cpp compiled with "tensor cores" support, which improves performance on NVIDIA RTX cards in most cases.
* **streamingllm**: Experimental feature to avoid re-evaluating the entire prompt when part of it is removed, for instance, when you hit the context length for the model in chat mode and an old message is removed.
* **cpu**: Force a version of llama.cpp compiled without GPU acceleration to be used. Can usually be ignored. Only set this if you want to use CPU only and llama.cpp doesn't work otherwise. 
* **no_mul_mat_q**: Disable the mul_mat_q kernel. This kernel usually improves generation speed significantly. This option to disable it is included in case it doesn't work on some system.
* **no-mmap**: Loads the model into memory at once, possibly preventing I/O operations later on at the cost of a longer load time.
* **mlock**: Force the system to keep the model in RAM rather than swapping or compressing (no idea what this means, never used it).
* **numa**: May improve performance on certain multi-cpu systems.

## Model dropdown

Here you can select a model to be loaded, refresh the list of available models (🔄), load/unload/reload the selected model, and save the settings for the model. The "settings" are the values in the input fields (checkboxes, sliders, dropdowns) below this dropdown. 

After saving, those settings will get restored whenever you select that model again in the dropdown menu.

If the **Autoload the model** checkbox is selected, the model will be loaded as soon as it is selected in this menu. Otherwise, you will have to click on the "Load" button.

## LoRA dropdown

Used to apply LoRAs to the model. Note that LoRA support is not implemented for all loaders. Check this [page](https://github.com/oobabooga/text-generation-webui/wiki) for details.

## Download model or LoRA

Here you can download a model or LoRA directly from the https://huggingface.co/ website.

* Models will be saved to `text-generation-webui/models`.
* LoRAs will be saved to `text-generation-webui/loras`.

In the input field, you can enter either the Hugging Face username/model path (like `facebook/galactica-125m`) or the full model URL (like `https://huggingface.co/facebook/galactica-125m`). To specify a branch, add it at the end after a ":" character like this: `facebook/galactica-125m:main`. 

To download a single file, as necessary for models in GGUF format, you can click on "Get file list" after entering the model path in the input field, and then copy and paste the desired file name in the "File name" field before clicking on "Download".
