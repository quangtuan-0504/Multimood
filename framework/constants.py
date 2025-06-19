CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100

# Image arguments
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# Video arguments
VIDEO_TOKEN_INDEX = 151647
DEFAULT_VIDEO_TOKEN = "<video>"
NUM_FRAMES = 8
MAX_FRAMES = 32
NUM_FRAMES_PER_SECOND = 1

# Audio arguments
AUDIO_TOKEN_INDEX = -202
DEFAULT_AUDIO_TOKEN = "<audio>"


# DAMO-NLP-SG/VideoLLaMA2.1-7B-AV
MODAL_INDEX_MAP = {
    "<image>": -200,
    "<video>": -201,
    "<audio>": -202,
    "<history>": -203
}


# Qwen2.5 Instruct
# MODAL_INDEX_MAP = {
#     # '<MEM>':151646,
#     '<history>':151665,
#     '<video>':151666,
#     '<MEM>':151667
# }

# google/gemma-2-9b-it
# MODAL_INDEX_MAP = {
#     '<history>':256002,
#     '<video>':256003,
#     '<MEM>':256004
# }


# # meta-llama/Llama-3.1-8B-Instruct
# MODAL_INDEX_MAP = {
#     '<history>':128258,
#     '<video>':128259,
#     '<MEM>':128260
# }