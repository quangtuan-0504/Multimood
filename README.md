# 🧠 MultiMood Fine-tuning & Evaluation Guide


This repository supports multiple workflows for fine-tuning and evaluating video-language models. Please choose the appropriate branch based on your training or evaluation objective:

![Overview](overview.png)

---

## 🔧 Fine-tuning Options

### 1. Supervised Fine-tuning
- **Branch:** [`feature/train_sft`](https://anonymous.4open.science/r/Multimood-C558/)
- **Anonymous Link:** https://anonymous.4open.science/r/Multimood-C558/
- **Description:** Use this branch if you want to perform supervised fine-tuning on the video-language model. This method relies on labeled training data to guide the learning process.

### 2. Fine-tuning with GRPO (Gradient-based Reinforcement Preference Optimization)
- **Branch:** [`feature/GRPO`](https://anonymous.4open.science/r/Multimood-BE8D/)
- **Anonymous Link:** https://anonymous.4open.science/r/Multimood-BE8D/
- **Description:** Use this branch to fine-tune the model using GRPO, a reinforcement learning technique that optimizes model behavior based on preference signals.

### 3. Fine-tuning with PPO (Proximal Policy Optimization)
- **Branch:** [`feature/PPO`](https://anonymous.4open.science/r/Multimood-D7BE/)
- **Anonymous Link:** https://anonymous.4open.science/r/Multimood-D7BE/
- **Description:** Use this branch for reinforcement learning via PPO. It is suitable for optimizing models based on reward functions when labeled data is limited or unavailable.

---

## 📊 Model Evaluation

### 4. Inference & Evaluation
- **Branch:** [`feature/inference_eval`](https://anonymous.4open.science/r/Multimood-0D26/)
- **Anonymous Link:** https://anonymous.4open.science/r/Multimood-0D26
- **Description:** Use this branch to evaluate the performance of your fine-tuned model. It includes inference scripts and evaluation metrics to benchmark model quality on various tasks.

---

### Acknowledge
- Special thanks to the **DAMO-NLP-SG Team**, whose work on the **VideoLLaMA** projects inspired this repository.
