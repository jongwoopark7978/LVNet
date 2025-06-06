# NOTE: Cannot download data...
# NOTE: Do I need to download checkpoint? for which stage?

# NOTE: What should be the video length?
# NOTE: How much of filter for each stage?

# base
# NOTE: Need to get this from Jongwoo.
dict_api = {
    "api_key":"ADD",
}
base_dir = "vima" # your data path ex) img_folder_path/egoschema


# scene clustering
divlam     = 12
# NOTE: What does ES dataset look like?
f_path     = f"{base_dir}/keywords.json" # files from keywords dir
q_path     = f"{base_dir}/questions.json" # files from questions dir
a_path     = f"{base_dir}/answer.json"
img_folder = f"{base_dir}/data" # your img folder path ex) img_folder_path/egoschema/frames_900_4531/q_uid/image_sec_millisec.jpg


# coarse key frame detector
maximgslen     = 32
limit_keywords = 25
concatname     = "LVnet"
modelpath      = "clippy_5k.pt" # model path
question_path  = f"{base_dir}/answer.json" # recommend using the same path with scene clustering answer path
answerpath     = f"{base_dir}/kwkfmatching/kf_{concatname}.jsonl"  # kwkfmatching is not necessary.
concatdir      = f"{base_dir}/kwkfmatching/concatimg_{concatname}" # kwkfmatching is not necessary.


# fine key frame detector
kf_vlm            = "gpt-4o"
kf_temp           = None
kf_num_select     = 3
kf_num_input_imgs = 32
kf_question_path  = f"{base_dir}/kwkfmatching/kf_{concatname}.jsonl" # recommend using the same path with coarse key frame detector answer path
kf_answer_path    = f"{base_dir}/kf_VLM/kf_VLM{kf_num_input_imgs}sel{kf_num_select}_{kf_question_path.split('/')[-1].split('.')[0]}.jsonl" # kf_VLM is not necessary.


# fine key frame detector refine
refine_num_group   = 4
refine_kflen       = 12
refine_output_path = f"{base_dir}/kf_VLM/refine/" + kf_answer_path.split('/')[-1] # kf_VLM is not necessary.
