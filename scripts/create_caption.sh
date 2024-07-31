# run this script from the root of the repo
# fix paths to data and update GPT keys in code

export PYTHONPATH=$PYTHONPATH:$PWD

python3 VLM_stage.py \
  --output-dir ego_base_link_bigtensor \
  --question-path your_question_path.jsonl \
  --gptmodel "gpt-4o" \
  --num-kf 12 \
  --temp 0
