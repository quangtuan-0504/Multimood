##### Run this after activate the environment #####
pip uninstall bitsandbytes -y
pip install bitsandbytes==0.43.1
pip install h5py rouge_score
pip install flash-attn --no-build-isolation
pip install --upgrade transformers
pip uninstall deepspeed -y
pip install deepspeed==0.15.4
pip install gdown
pip install git+https://github.com/huggingface/trl.git
pip install --upgrade peft

