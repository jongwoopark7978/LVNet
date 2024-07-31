"""
Evaluate a model on the egoschema dataset using LVNet (captions pre-generated)

Sample Run:

python3 LLM_stage.py \
  --output-dir ego_base_link \
  --captions data/ES_captions_gpt4o.jsonl \
  --per-vid-captions 12 \
  --gptmodel "gpt-4o" \
  --temperature 0.0
"""

import argparse
import json
import os

from tqdm import tqdm

from src.run_gpt import run_gpt

# You may add multiple keys to run parallel calls
dict_api = {
    "api_key": "ADD",
}


_PROMPT_TEMPLATE = (
    "Here are descriptions of the video frames at specific times, noted in seconds."
    "\n\n{Putdesc}.\n\nThe descriptions of the frames conclude. Think step-by-step"
    " and I request your selection of the most appropriate response to the following"
    " question\n\nQuestion:\n{Putquestion}\n\nOptions:\n{AllOptions}"
)


def eval_model(args):
    # change split to split
    captions_path, data_path, split, gptmodel, temp, base_dir, job_name = (
        args.captions,
        args.data,
        args.per_vid_captions,
        args.gptmodel,
        args.temperature,
        args.output_dir,
        args.job_name,
    )

    prompt = _PROMPT_TEMPLATE

    os.makedirs(base_dir, exist_ok=True)
    output_dir = f"{base_dir}/egoschema/{job_name}"
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    save_name = captions_path.rsplit("/", 2)[-1].replace(".jsonl", "")
    output_summary_path = f"{output_dir}/{save_name}.jsonl"
    print(f"Saving outputs to:{output_summary_path}")
    output_summary = open(output_summary_path, "w")

    input_summary = [
        json.loads(q) for q in open(os.path.expanduser(captions_path), "r")
    ]
    dataset = json.load(open(os.path.expanduser(data_path), "r"))
    input_len = len(input_summary)
    assert (
        input_len % split == 0
    ), f"input_len%split:{input_len%split}, input_len:{input_len}, split:{split}"
    groups = input_len // split

    final_prompts = []
    final_info = []
    for i in tqdm(range(groups)):
        sidx = i * split
        eidx = (i + 1) * split

        desc = ""
        timeline = []
        for idx, e in enumerate(input_summary[sidx:eidx]):
            cur_data = dataset[e["q_uid"]]
            desc += e["answer"] + " "
            timeline.append(e["timeline"])

            if idx == split - 1:  # the last of split
                action_0 = cur_data["option 0"]
                action_1 = cur_data["option 1"]
                action_2 = cur_data["option 2"]
                action_3 = cur_data["option 3"]
                action_4 = cur_data["option 4"]

                option_list = ""
                option_number_candidate = ["one", "two", "three", "four", "five"]
                option_number = option_number_candidate[4]
                AllOptNumber = "option 0, option 1, option 2, option 3, option 4"
                FocusOptions = ""

                alloptions = f"option 0: {action_0}\noption 1: {action_1}\noption 2: {action_2}\noption 3: {action_3}\noption 4: {action_4}"
                option_list = f"option 0: {action_0}\noption 1: {action_1}\noption 2: {action_2}\noption 3: {action_3}\noption 4: {action_4}"

                FocusOptions += "option 0, option 1, option 2, option 3, option 4"

                question = cur_data["question"]

                curr_prompt = (
                    prompt.replace("{Putdesc}", desc)
                    .replace("{Putquestion}", question)
                    .replace("{Putoptions}", option_list)
                    .replace("{PutOptNumber}", option_number)
                    .replace("{FocusOptions}", FocusOptions)
                    .replace("{AllOptions}", alloptions)
                    .replace("{PutAllOptNumber}", AllOptNumber)
                )

                final_prompts.append(curr_prompt)

                CA_option = {}
                if "CA" in cur_data:
                    CA_option = {"CA": cur_data["CA"]}

                info = {
                    "q_uid": e["q_uid"],
                    "prompt": curr_prompt,
                    "timeline": timeline,
                    "question": question,
                    "option 0": action_0,
                    "option 1": action_1,
                    "option 2": action_2,
                    "option 3": action_3,
                    "option 4": action_4,
                } | CA_option

                final_info.append(info)

    output_VLM = run_gpt(
        texts=final_prompts,
        api_keys=list(dict_api.values()),
        max_tokens=2000,
        model=gptmodel,
        temperature=temp,
        num_threads=20,  # Tune this
        backoff_time=1 * 60,
        silent=False,
        dataset="egoschema",
    )

    output_VLM = list(output_VLM)

    for q_idx, info in enumerate(tqdm(final_info)):  # prompt_list = # Q&C
        info["answer"] = output_VLM[q_idx]
        output_summary.write(json.dumps(info) + "\n")

    # finish the summarization for the current question
    output_summary.close()
    print(f"output_summary_path:{output_summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--job-name", type=str, default="run001")
    parser.add_argument("--captions", type=str, default="data/ES_captions_gpt4o.jsonl")
    parser.add_argument("--data", type=str, default="data/ES_qa_data.json")
    parser.add_argument("--per-vid-captions", type=int, default=12)
    parser.add_argument("--gptmodel", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--temperature", type=float, default=None)

    args = parser.parse_args()

    eval_model(args)
