import transformers
from .train import DataCollatorForSupervisedDataset, LazySupervisedDataset, DataArguments


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Create dataset and data collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args
    )

    # LazySupervisedDataset loads data including CSV + video + audio
    # Output: nn.Dataloader compatible dataset
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )


# Initialize tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV",
    # Optional: specify cache directory to store downloaded model/tokenizer files
    # cache_dir="",
    model_max_length=512,
    padding_side="right",
    use_fast=True,
)

# Add special tokens for video, audio, memory, and history context
tokenizer.add_special_tokens({
    'additional_special_tokens': [
        '<|im_start|>', '<|im_end|>', '<MEM>', '<history>',
        '<kVidHis>', '<kAudHis>', '<video>', '<audio>'
    ]
})


# Build supervised data module
data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)