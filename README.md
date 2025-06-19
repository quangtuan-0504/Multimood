# 🧠 MultiMood - Model Evaluation on the Test Set

MultiMood is a multimodal natural language processing system (audio and visual) capable of fine-tuning and evaluating models on a provided dataset. This guide walks you through setting up the environment, running inference, and evaluating the model on the **test** set.

---

## 💥 System Requirements

- **CUDA**: 12.4  
- **Python**: ≥ 3.10  
- **Conda**: Recommended to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)

---

## 🧹 Step 1: Clone the source code and switch to the `feature/inference_eval` branch

```bash
git clone <link-to-github-repo>
cd MultiMood
git checkout feature/inference_eval
```

> Replace `<link-to-github-repo>` with the actual GitHub repository URL.

---

## 🧪 Step 2: Create a virtual environment and install base dependencies

```bash
bash vast_setup_env.sh
```

> If your system encounters issues with Python versions, you can modify the `environment.yml` file to set `python=3.10` or another compatible version.

---

## ⚙️ Step 3: Manually install required libraries

```bash
# Reinstall bitsandbytes for compatibility
pip uninstall bitsandbytes -y
pip install bitsandbytes==0.43.1

# Training libraries
pip install h5py
pip install flash-attn --no-build-isolation

# Fix DeepSpeed compatibility
pip uninstall deepspeed -y
pip install deepspeed==0.15.4

# Upgrade Transformers and NLP-related libraries
pip install --upgrade transformers==4.51.1
pip install scikit-learn==1.6.1
pip install nltk==3.9.1
pip install rouge_score==0.1.2
pip install python-dotenv==0.9.0
pip install bert_score==0.3.13
pip install autotune==0.0.3
```

---

## 📥 Step 4: Download training and evaluation datasets

```bash
bash download_data.sh
```

> This script downloads the training and validation datasets and prepares necessary paths for training or inference.

---

## 🧪 Step 5: Run inference on the Test Set

```bash
python evaluation_smes/inference_smes.py   --dataset-path <path_to_jsonl_test_set>   --model-path <model_name_or_path>   --modal-type <a_or_v_or_av>   --res-folder <path_to_file_result_inference>
```

### Arguments:
- `--dataset-path`: Path to the JSONL file containing the test set.
- `--model-path`: Path to the fine-tuned model checkpoint.
- `--modal-type`: Input data type (`a` = audio, `v` = video, `av` = audio + video).
- `--res-folder`: Folder to store the inference results.

---

## 📊 Step 6: Evaluate the results on the Test Set

```bash
python evaluation_smes/eval_final.py <path_to_file_result_inference> -o <path_to_file_report>
```

### Arguments:
- `path_to_file_result_inference`: Inference output from Step 5.
- `-o`: Path to save the evaluation report.

---

## ✅ Output

After completing the steps above, you will obtain:
- Inference result file (usually `.jsonl` or `.json`)
- Evaluation report file (`.txt`, `.csv`, or `.json`) containing metrics such as accuracy, F1-score, BLEU, etc., depending on the script configuration.

---

## 📬 Contact

For any issues regarding installation or running the project, please create an issue on the project's GitHub repository.
