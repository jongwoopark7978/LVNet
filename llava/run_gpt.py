import requests
from time import sleep
from rp import (
    encode_image_to_base64,
    is_image,
    lazy_par_map,
    fansi_print,
    format_current_date,
)

import rp


def _get_gpt_request_json(image, text, max_tokens, model, temperature, dataset="egoschema"):

    options = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    options = {key: value for key, value in options.items() if value is not None}

    sysprompt = ""
    if dataset == "egoschema":
        sysprompt = 'You are a language model designed to analyze video content based on provided captions. These captions summarize key information from multiple frames. In the captions, \'C\' stands for the cameraman. \nYour task is to use these captions to answer questions about the video. For each question, you will choose the most accurate answer from five options, relying only on the given captions without any external knowledge. Provide your response following this specified JSON format.\n{"selection": write your selection number here, "reasons": state the reasons for your selection less than 30 words.}. This is one example output format. {"selection": 3, "reasons": The primary objective and focus within the video content is on cleaning dishes}'
    else:
        sysprompt = 'You are a language model designed to analyze video content based on provided captions. These captions summarize key information from multiple frames.\nYour task is to use these captions to answer questions about the video. For each question, you will choose the most accurate answer from five options, relying only on the given captions without any external knowledge. Provide your response following this specified JSON format.\n{"selection": write your selection number here, "reasons": state the reasons for your selection less than 30 words.}. This is one example output format. {"selection": 3, "reasons": The primary objective and focus within the video content is on cleaning dishes}'

    if image == None:
        context = {"role": "user",
                   "content": [{"type": "text","text": text,},],
                  }

    else:
        base64_image = encode_image_to_base64(image)
        context = {"role": "user",
                   "content": [{"type": "text", "text": text,},
                               {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},},
                              ],
                  }

    return options | {
        "model": model,
        "messages": [
            {"role": "system", "content": sysprompt},
            context,
        ],
    }

def _run_gpt(image, text, max_tokens, model, temperature, api_key):
    """Processes a single text"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    request_json = _get_gpt_request_json(image, text, max_tokens, model, temperature)

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=request_json,
    )

    return response.json()["choices"][0]["message"]["content"]

def run_gpt(
    images=None,
    texts="",
    api_keys=None,
    max_tokens=2000,
    model="gpt-4-vision-preview",
    temperature=None,
    num_threads=10,
    backoff_time=1 * 60,
    silent=False,
    dataset="egoschema",
    verbose=True,
):
    """
    Asks GPT a question about an text, returning a string.
    If given multiple texts, will process them in parallel lazily (retuning a generator instead)

    Args:
        text (str, optional): The question we ask GPT
        max_tokens (int, optional): Maximum tokens in the response
        api_key (str, list[str], optional): If specified, overwrites the default openai api_key
            Can be a string or a list of strings
        backoff_time (float): number of seconds to wait if there's an error
        num_threads (int): number of threads if you'd like to run in parallel. 1 thread means it runs sequentially
        silent (bool): if True, won't report errors
        temperature (float, optional)
        model (str)

    Returns:
        (str or generator): GPT3.5, GPT4's response (or a lazy generator of responses if given a list of texts)
    """
    assert api_keys is not None, "Please provide api_keys for GPT calls"
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    if isinstance(texts, str):
        texts = [texts]
    if images is None: images = [None]
    elif isinstance(images, str) or is_image(images): images = [images]

    assert len(images)==len(texts) or len(images)==1 or len(texts)==1

    if len(texts)==1: texts = texts * len(images)
    if len(images)==1: images = list(images) * len(texts)

    assert len(images) == len(texts)

    api_key_index = 0

    def run(args):
        image, text = args

        while True:
            nonlocal api_key_index
            api_key_index += 1
            api_key_index %= len(api_keys)

            api_key = api_keys[api_key_index]

            try:
                output = _run_gpt(image, text, max_tokens, model, temperature, api_key)
                if verbose: fansi_print("output: " + str(output), "yellow", "bold")
                return output
            except Exception as e:
                if not silent:
                    # rp.print_stack_trace()
                    # rp.print_verbose_stack_trace()
                    fansi_print(
                        "Error (" + format_current_date() + "): " + repr(e),
                        "red",
                        "bold",
                    )
                sleep(backoff_time)

    return lazy_par_map(
        run,
        zip(images, texts),
        num_threads=num_threads,
    )
