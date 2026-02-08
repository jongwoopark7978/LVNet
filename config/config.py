# base
dict_api = {
    "api_key":"ADD",
}
base_dir = "your_path" # your data path ex) img_folder_path/egoschema


# scene clustering
divlam     = 12
f_path     = "your_path" # reorg QC filter json path
q_path     = "your_path" # init_question json path
a_path     = "your_path"
img_folder = "your_path" # your img folder path ex) img_folder_path/egoschema/frames_900_4531/q_uid/image_sec_millisec.jpg


# coarse key frame detector
maximgslen     = 32
limit_keywords = 25
concatname     = "LVnet"
modelpath      = "your_path" # model path
question_path  = "your_path" # recommend using the same path with scene clustering answer path
answerpath     = f"{base_dir}/kwkfmatching/kf_{concatname}.jsonl"  # kwkfmatching is not necessary.
concatdir      = f"{base_dir}/kwkfmatching/concatimg_{concatname}" # kwkfmatching is not necessary.


# fine key frame detector
kf_vlm            = "gpt-4o"
kf_temp           = None
kf_num_select     = 3
kf_num_input_imgs = 32
kf_question_path  = "your_path" # recommend using the same path with coarse key frame detector answer path
kf_answer_path    = f"{base_dir}/kf_VLM/kf_VLM{kf_num_input_imgs}sel{kf_num_select}_{kf_question_path.split('/')[-1].split('.')[0]}.jsonl" # kf_VLM is not necessary.


# fine key frame detector refine
refine_num_group   = 4
refine_kflen       = 12
refine_output_path = f"{base_dir}/kf_VLM/refine/" + kf_answer_path.split('/')[-1] # kf_VLM is not necessary.
