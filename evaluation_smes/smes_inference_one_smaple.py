import sys
import os
sys.path.append('/')
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))
from framework import model_init, mm_infer
from framework.utils import disable_torch_init
import argparse

def inference(args):

    model_path = args.model_path
    model, processor, tokenizer = model_init(model_path)
    # Initialize dataloader
    # Extract (prompt, video path) from each sample, run inference, and save prediction and label
    # Loop through each sample below
    if args.modal_type == "a":
        model.model.vision_tower = None
    elif args.modal_type == "v":
        model.model.audio_tower = None
    elif args.modal_type == "av":
        pass
    else:
        raise NotImplementedError
    # Audio-visual Inference
    audio_video_path = "audio_video_path.mp4"
    preprocess = processor['audio' if args.modal_type == "a" else "video"]
    if args.modal_type == "a":
        audio_video_tensor = preprocess(audio_video_path)
    else:
        audio_video_tensor = preprocess(audio_video_path, va=True if args.modal_type == "av" else False)
    question = f"Describe the emotions of the person in the blue shirt in the video"

    output = mm_infer(
        audio_video_tensor,
        question,
        model=model,
        tokenizer=tokenizer,
        modal='audio' if args.modal_type == "a" else "video",
        do_sample=False,
    )

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=False, default='DAMO-NLP-SG/VideoLLaMA2.1-7B-AV')
    parser.add_argument('--modal-type', choices=["a", "v", "av"], help='', default='av')
    args = parser.parse_args()

    print('answer : ',inference(args))