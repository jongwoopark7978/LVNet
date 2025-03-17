import argparse
import json
import os

from tqdm import tqdm
from src.run_gpt import run_gpt

"""
Extract keywords from the given question and options

Sample Run
python3 extractKeyword.py --output-dir ego_base_link --question questions/500questions.jsonl --gptmodel "gpt-4-1106-preview"

"""


# You may add multiple keys if you want parallel calls
dict_api = {
    "api_key": "ADD",
}

PROMPT = (
    "Think step-by-step and for each option, identify all the specified activities. "
    "Each description of activity should use active voice with plain verbs, contain fewer than six words, "
    "and retains as many original terms from the options as possible.\n"
    "Here are the options:\n\n"
    "option 0: {Putop0}\n"
    "option 1: {Putop1}\n"
    "option 2: {Putop2}\n"
    "option 3: {Putop3}\n"
    "option 4: {Putop4}\n"
    "option 5: {Putquestion}.\n"
    "All the options were introduced. 'C' represents the camera operator in the options.  "
    "Your answer should follow the JSON format shown below and should only include the JSON result. "
    "Do not output any warnings or notes under any circumstances. Instead, adhere strictly to the provided JSON format example.\n"
    "This is one example output format.\n"
    "{\"option 0\": [\"plays soccer\", \"go to school\"], \"option 1\": [\"go to the gym\", \"go to school\"], "
    "\"option 2\": [\"go to school\", \"dry hair\"], \"option 3\": [\"plays basketball\", \"look at the tree\"], "
    "\"option 4\": [\"plays soccer\", \"drop the ball\"], \"option 5\": [\"turn the table\", \"go to school\"]}"
)


def main(args):
    # 1. Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    job_dir = os.path.join(args.output_dir, "extractedKeywords")
    os.makedirs(job_dir, exist_ok=True)


    # 2. Build the output file name (based on --question)
    question_file_name = os.path.basename(args.question).replace(".jsonl", "")
    output_summary_path = os.path.join(job_dir, f"{question_file_name}.jsonl")
    print(f"Saving outputs to: {output_summary_path}")

    # 3. Read the question file
    with open(os.path.expanduser(args.question), "r") as f:
        question_data = [json.loads(line) for line in f]

    # 4. Construct final prompts
    final_prompts = []
    final_info = []
    for entry in tqdm(question_data, desc="Building prompts"):
        q_uid = entry["q_uid"]
        # Insert each option + question into the embedded prompt
        cur_prompt = (
            PROMPT
            .replace("{Putop0}", entry["option 0"])
            .replace("{Putop1}", entry["option 1"])
            .replace("{Putop2}", entry["option 2"])
            .replace("{Putop3}", entry["option 3"])
            .replace("{Putop4}", entry["option 4"])
            .replace("{Putquestion}", entry["question"])
        )

        final_prompts.append(cur_prompt)

        # Track data for JSON output
        info = {
            "q_uid": q_uid,
            "prompt": cur_prompt,
            "option 0": entry["option 0"],
            "option 1": entry["option 1"],
            "option 2": entry["option 2"],
            "option 3": entry["option 3"],
            "option 4": entry["option 4"],
            "question": entry["question"],
        }

        # Include ground-truth label if present
        if "CA" in entry:
            info["CA"] = entry["CA"]

        final_info.append(info)

    # 5. Call GPT
    print("Sending prompts to GPT. This may take a while...")
    output_results = run_gpt(
        texts=final_prompts,
        api_keys=list(dict_api.values()),
        max_tokens=2000,
        model=args.gptmodel,
        temperature=args.temperature,
        num_threads=5,    # Adjust as needed
        backoff_time=60,   # Adjust as needed
        silent=False,
        dataset="extractKeyword",
    )

    output_results = list(output_results)

    # 6. Save results
    with open(output_summary_path, "w") as outfile:
        for i, info in enumerate(final_info):
            info["answer"] = output_results[i]
            outfile.write(json.dumps(info) + "\n")

    print(f"Done! Results written to {output_summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to store the resulting JSONL file.")
    parser.add_argument("--question", type=str, required=True,
                        help="Path to the JSONL file with question data (e.g., 500questions.jsonl).")
    parser.add_argument("--gptmodel", type=str, default="gpt-4-1106-preview",
                        help="The GPT model to call.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature parameter for GPT.")

    args = parser.parse_args()
    main(args)
