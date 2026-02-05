# MultiMood - Fine-tuning with GRPO (Guided Reinforcement Preference Optimization)

This guide provides detailed steps to set up the environment, install dependencies, serve the reward function, download the dataset, and train the model using **GRPO** (Guided Reinforcement Preference Optimization).

## 💥 System Requirements

- CUDA: **12.4**
- Python: **≥ 3.10** (Python 3.12 is used for reward server)
- Conda (Recommended: Miniconda or Anaconda)
- GPU with support for `torchrun` (multi-GPU training)

---

## 🧹 Step 1: Clone the source code and switch to the GRPO branch

```bash
git clone https://github.com/quangtuan-0504/Multimood/
cd Multimood
git checkout feature/GRPO
```

---

## 🧪 Step 2: Create the virtual environment and install base dependencies

```bash
conda env create -f environment.yml
conda activate grpo
```

> If your system has issues with Python version conflicts, you may modify `environment.yml` to target a compatible Python version (e.g., `python=3.10`).

---

## ⚙️ Step 3: Install additional required libraries manually

```bash
# Fix compatibility for bitsandbytes
pip uninstall bitsandbytes -y
pip install bitsandbytes==0.43.1

# Install required libraries for training
pip install h5py
pip install flash-attn --no-build-isolation

# Fix DeepSpeed compatibility
pip uninstall deepspeed -y
pip install deepspeed==0.15.4

# Update Transformers and NLP-related libraries
pip install --upgrade transformers==4.51.1
pip install scikit-learn==1.6.1
pip install nltk==3.9.1
pip install rouge_score==0.1.2
pip install python-dotenv==0.9.0
pip install bert_score==0.3.13
pip install FlagEmbedding==1.3.4
pip install autotune==0.0.3
```

---

## 📥 Step 4: Download the dataset

Use the provided script to download all necessary training and validation data.

```bash
bash download_data.sh
```

> This will generate datasets including the path needed in the training command.

---

## 🛠 Step 5: Serve the reward function (BGEM3)

To run the reward model as a service, set up a separate environment:

```bash
# Create new environment for the reward model service
conda create -n service_bgem3 python=3.12 -y
conda activate service_bgem3

# Install dependencies
pip install -r requirement_service_bgem3.txt

# Start the reward service
python serving_bgem3/service.py
```

> Ensure this service is running **in parallel** when training the model with GRPO.

---

## 🚀 Step 6: Train the model using GRPO

Use the following command to start training. Modify parameters if needed:

```bash
torchrun --nproc_per_node=2 GRPO/train_base.py \
  --dataset-path DATA_PATH \
  --model-id 'MODEL_PATH'
```

### 🧷 Arguments explained:
- `--nproc_per_node=2`: Number of GPUs to use. Set to `1` if using a single GPU.
- `--dataset-path`: Path to the `.jsonl` dataset file downloaded in Step 4.
- `--model-id`: Hugging Face model name.

---

## 📌 Notes

- Make sure all CUDA-dependent packages (like `flash-attn`, `deepspeed`, `bitsandbytes`) are compiled/installed with CUDA 12.4 compatibility.
- If you face issues with Flash Attention or other compiled packages, refer to their specific build instructions for CUDA 12.4.
---


## ✅ Done!

You’re ready to fine-tune using GRPO!  
If you encounter any issues during setup or training, please refer to this guide again or raise an issue in the [GitHub Repository].

