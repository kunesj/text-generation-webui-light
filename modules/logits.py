import time
import traceback

import numpy as np

from modules import models, shared
from modules.models import load_model
from modules.utils import check_model_loaded

global_scores = None


def get_next_logits(*args, **kwargs):
    if shared.args.idle_timeout > 0 and shared.model is None and shared.model_name not in [None, "None"]:
        shared.model, shared.tokenizer = load_model(shared.model_name)

    needs_lock = not args[2]  # use_samplers
    if needs_lock:
        shared.generation_lock.acquire()

    try:
        result = _get_next_logits(*args, **kwargs)
    except Exception:
        traceback.print_exc()
        result = None

    if needs_lock:
        models.last_generation_time = time.time()
        shared.generation_lock.release()

    return result


def _get_next_logits(prompt, state, use_samplers, previous, top_logits=25, return_dict=False):
    model_is_loaded, error_message = check_model_loaded()
    if not model_is_loaded:
        return error_message, previous

    # llama.cpp case
    if shared.model.__class__.__name__ == "LlamaServer":
        logprobs = shared.model.get_logits(prompt, state, n_probs=top_logits, use_samplers=use_samplers)

        if return_dict:
            output = {}
            for entry in logprobs:
                token = repr(entry["token"])
                if len(token) > 2 and token.startswith("'") and token.endswith("'"):
                    token = token[1:-1]

                prob = entry["prob"] if use_samplers else np.exp(entry["logprob"])
                output[token] = prob
            return output
        output = ""
        for entry in logprobs:
            token = repr(entry["token"])
            if len(token) > 2 and token.startswith("'") and token.endswith("'"):
                token = token[1:-1]

            prob = entry["prob"] if use_samplers else np.exp(entry["logprob"])
            output += f"{prob:.5f}  -  {token}\n"
        return output, previous

    # All other model types
    raise NotImplementedError()
