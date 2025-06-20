import transformers
from .train import DataCollatorForSupervisedDataset, LazySupervisedDataset, DataArguments


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args
    )
    # nn.Dataset load csv + video + audio
    # -> nn.Dataloader
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


# Initialize tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV",
    # cache_dir ="",# chỉ định thư mục cache để lưu
    model_max_length=512,
    padding_side="right",
    use_fast=True,
)

# add token video , audio , mambar
tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<MEM>','<history>',
            '<kVidHis>', '<kAudHis>' , '<video>', '<audio>']})



data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)