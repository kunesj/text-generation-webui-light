import os
import time
import pathlib
from typing import Any

from modules import shared
from modules.logging_colors import logger
from modules.models_settings import get_model_metadata
from modules.utils import resolve_model_path

last_generation_time = time.time()


def load_model(model_name, loader=None):
    logger.info(f'Loading "{model_name}"')
    t0 = time.time()

    shared.is_seq2seq = False
    shared.model_name = model_name
    load_func_map = {
        "llama.cpp": llama_cpp_server_loader,
    }

    metadata = get_model_metadata(model_name)
    if loader is None:
        if shared.args.loader is not None:
            loader = shared.args.loader
        else:
            loader = metadata["loader"]
            if loader is None:
                logger.error("The path to the model does not exist. Exiting.")
                raise ValueError

    shared.args.loader = loader
    output = load_func_map[loader](model_name)
    if type(output) is tuple:
        model, tokenizer = output
    else:
        raise NotImplementedError()

    if model is None:
        return None, None

    shared.settings.update({k: v for k, v in metadata.items() if k in shared.settings})
    if loader == "llama.cpp":
        shared.settings["truncation_length"] = shared.args.ctx_size

    shared.is_multimodal = False
    if loader.lower() in ("llama.cpp",) and hasattr(model, "is_multimodal"):
        shared.is_multimodal = model.is_multimodal()

    logger.info(f'Loaded "{model_name}" in {(time.time() - t0):.2f} seconds.')
    logger.info(f'LOADER: "{loader}"')
    logger.info(f"TRUNCATION LENGTH: {shared.settings['truncation_length']}")
    logger.info(f'INSTRUCTION TEMPLATE: "{metadata["instruction_template"]}"')
    return model, tokenizer


def llama_cpp_server_loader(model_name: str) -> tuple[Any, Any]:
    from modules.llama_cpp_server import LlamaServer

    path = resolve_model_path(model_name)

    if path.is_file():
        model_file = path
    else:
        gguf_files = sorted(path.glob("*.gguf"))
        if not gguf_files:
            logger.error(f"No .gguf models found in the directory: {path}")
            return None, None

        model_file = gguf_files[0]

    if raw_server_path := os.getenv("LLAMA_SERVER_PATH"):
        server_path = pathlib.Path(raw_server_path).absolute()

    else:
        logger.error("llama-server not specified")
        return None, None

    try:
        model = LlamaServer(model_file, server_path)
        return model, model
    except Exception as e:
        logger.error("Error loading the model with llama.cpp: %s", e)
        return None, None


def unload_model(keep_model_name=False):
    if shared.model is None:
        return

    model_class_name = shared.model.__class__.__name__
    is_llamacpp = model_class_name == "LlamaServer"

    shared.model = shared.tokenizer = None
    shared.lora_names = []
    shared.model_dirty_from_training = False

    if not is_llamacpp:
        from modules.torch_utils import clear_torch_cache

        clear_torch_cache()

    if not keep_model_name:
        shared.model_name = "None"


def reload_model():
    unload_model()
    shared.model, shared.tokenizer = load_model(shared.model_name)


def unload_model_if_idle():
    global last_generation_time

    logger.info(f"Setting a timeout of {shared.args.idle_timeout} minutes to unload the model in case of inactivity.")

    while True:
        shared.generation_lock.acquire()
        try:
            if time.time() - last_generation_time > shared.args.idle_timeout * 60:
                if shared.model is not None:
                    logger.info("Unloading the model for inactivity.")
                    unload_model(keep_model_name=True)
        finally:
            shared.generation_lock.release()

        time.sleep(60)
