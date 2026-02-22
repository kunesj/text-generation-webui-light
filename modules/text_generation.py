import ast
import copy
import html
import random
import time
import traceback

import numpy as np

from modules import models, shared
from modules.extensions import apply_extensions
from modules.html_generator import generate_basic_html
from modules.logging_colors import logger
from modules.utils import check_model_loaded


def generate_reply(*args, **kwargs):
    if shared.args.idle_timeout > 0 and shared.model is None and shared.model_name not in [None, "None"]:
        from modules.models import load_model

        shared.model, shared.tokenizer = load_model(shared.model_name)

    shared.generation_lock.acquire()
    try:
        for result in _generate_reply(*args, **kwargs):
            yield result
    finally:
        models.last_generation_time = time.time()
        shared.generation_lock.release()


def _generate_reply(question, state, stopping_strings=None, is_chat=False, escape_html=False, for_ui=False):
    # Find the appropriate generation function
    generate_func = apply_extensions("custom_generate_reply")
    if generate_func is None:
        model_is_loaded, error_message = check_model_loaded()
        if not model_is_loaded:
            yield ""
            return

        if shared.model.__class__.__name__ in ["LlamaServer"]:
            generate_func = generate_reply_custom
        else:
            raise NotImplementedError()

    if shared.args.verbose:
        logger.info("PROMPT=")
        print_prompt(question)

    # Prepare the input
    original_question = question
    if not is_chat:
        state = apply_extensions("state", state)
        question = apply_extensions("input", question, state)

    # Find the stopping strings
    all_stop_strings = []
    for st in (stopping_strings, state["custom_stopping_strings"]):
        if type(st) is str:
            st = ast.literal_eval(f"[{st}]")

        if type(st) is list and len(st) > 0:
            all_stop_strings += st

    shared.stop_everything = False
    reply = ""
    is_stream = state["stream"]
    if len(all_stop_strings) > 0 and not state["stream"]:
        state = copy.deepcopy(state)
        state["stream"] = True

    # Generate
    last_update = -1
    latency_threshold = 1 / 1000
    for reply in generate_func(question, original_question, state, stopping_strings, is_chat=is_chat):
        cur_time = time.monotonic()
        reply, stop_found = apply_stopping_strings(reply, all_stop_strings)
        if escape_html:
            reply = html.escape(reply)

        if is_stream:
            # Limit number of tokens/second to make text readable in real time
            if state["max_tokens_second"] > 0:
                diff = 1 / state["max_tokens_second"] - (cur_time - last_update)
                if diff > 0:
                    time.sleep(diff)

                last_update = time.monotonic()
                yield reply

            # Limit updates to avoid lag in the Gradio UI
            # API updates are not limited
            else:
                # If 'generate_func' takes less than 0.001 seconds to yield the next token
                # (equivalent to more than 1000 tok/s), assume that the UI is lagging behind and skip yielding
                if (cur_time - last_update) > latency_threshold:
                    yield reply
                last_update = time.monotonic()

        if stop_found or (state["max_tokens_second"] > 0 and shared.stop_everything):
            break

    if not is_chat:
        reply = apply_extensions("output", reply, state)

    yield reply


def encode(prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
    if shared.tokenizer is None:
        raise ValueError("No tokenizer is loaded")

    # llama.cpp case
    if shared.model.__class__.__name__ == "LlamaServer":
        input_ids = shared.tokenizer.encode(str(prompt), add_bos_token=add_bos_token)
        input_ids = np.array(input_ids).reshape(1, len(input_ids))

        if truncation_length is not None:
            input_ids = input_ids[:, -truncation_length:]

        return input_ids

    # All other model types
    raise NotImplementedError()


def decode(output_ids, skip_special_tokens=True):
    if shared.tokenizer is None:
        raise ValueError("No tokenizer is loaded")

    return shared.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens)


def get_encoded_length(prompt):
    length_after_extensions = apply_extensions("tokenized_length", prompt)
    if length_after_extensions is not None:
        return length_after_extensions

    return len(encode(prompt)[0])


def get_token_ids(prompt):
    tokens = encode(prompt)[0]
    decoded_tokens = [shared.tokenizer.decode([int(i)]) for i in tokens]

    output = ""
    for row in list(zip(tokens, decoded_tokens, strict=False)):
        output += f"{str(int(row[0])).ljust(5)}  -  {row[1]!r}\n"

    return output


def get_max_prompt_length(state):
    return state["truncation_length"] - state["max_new_tokens"]


def generate_reply_wrapper(question, state, stopping_strings=None):
    """
    Returns formatted outputs for the UI
    """
    reply = question if not shared.is_seq2seq else ""
    yield formatted_outputs(reply, shared.model_name)

    for reply in generate_reply(question, state, stopping_strings, is_chat=False, escape_html=True, for_ui=True):
        if not shared.is_seq2seq:
            reply = question + reply

        yield formatted_outputs(reply, shared.model_name)


def formatted_outputs(reply, model_name):
    return html.unescape(reply), generate_basic_html(reply)


def set_manual_seed(seed):
    seed = int(seed)
    if seed == -1:
        seed = random.randint(1, 2**31)
    return seed


def stop_everything_event():
    shared.stop_everything = True


def apply_stopping_strings(reply, all_stop_strings):
    stop_found = False
    for string in all_stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou:"
        # is completed, trim it
        for string in all_stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
            else:
                continue

            break

    return reply, stop_found


def generate_reply_custom(question, original_question, state, stopping_strings=None, is_chat=False):
    """
    For models that do not use the transformers library for sampling
    """
    state = copy.deepcopy(state)
    state["seed"] = set_manual_seed(state["seed"])
    t0 = time.time()
    reply = ""
    try:
        if not is_chat:
            yield ""

        if not state["stream"]:
            reply = shared.model.generate(question, state)
            yield reply
        else:
            for reply in shared.model.generate_with_streaming(question, state):
                yield reply

    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()

        if hasattr(shared.model, "last_prompt_token_count"):
            original_tokens = shared.model.last_prompt_token_count
            new_tokens = len(encode(reply)[0]) if reply else 0
        else:
            original_tokens = len(encode(original_question)[0])
            new_tokens = len(encode(original_question + reply)[0]) - original_tokens

        logger.info(
            f"Output generated in {(t1 - t0):.2f} seconds ({new_tokens / (t1 - t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {state['seed']})"
        )
        return


def print_prompt(prompt, max_chars=-1):
    DARK_YELLOW = "\033[38;5;3m"
    RESET = "\033[0m"

    if max_chars > 0 and len(prompt) > max_chars:
        half_chars = max_chars // 2
        hidden_len = len(prompt[half_chars:-half_chars])
        hidden_msg = f"{DARK_YELLOW}[...{hidden_len} characters hidden...]{RESET}"
        print(prompt[:half_chars] + hidden_msg + prompt[-half_chars:])
    else:
        print(prompt)

    print()
