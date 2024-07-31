# run this script from the root of the repo
# fix paths to data and update GPT keys in code

export PYTHONPATH=$PYTHONPATH:$PWD

# Evaluate on ES subset (uncomment it if you want to run it)
python3 LLM_stage.py \
  --output-dir ego_base_link \
  --captions data/ESsub_captions_gpt4o.jsonl \
  --data data/ESsub_qa_data.json \
  --per-vid-captions 12 \
  --gptmodel "gpt-4o" \
  --temperature 0.0


# Evaluate on ES full dataset (uncomment it if you want to run it)
# python3 LLM_stage.py \
#   --output-dir ego_base_link \
#   --captions data/ES_captions_gpt4o.jsonl \
#   --data data/ES_qa_data.json \
#   --per-vid-captions 12 \
#   --gptmodel "gpt-4o" \
#   --temperature 0.0