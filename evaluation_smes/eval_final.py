import json
import pandas as pd
import re
import numpy as np
import argparse
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# NLP evaluation libraries
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score


def process_jsonl(file_path):
    """
    Process JSONL file and extract emotions, strategies and responses.
    Returns a DataFrame with all extracted data.
    """
    # Lists to store the extracted data
    label_client_emotions = []
    label_therapist_emotions = []
    label_therapist_strategies = []
    label_therapist_responses = []

    predict_client_emotions = []
    predict_therapist_emotions = []
    predict_therapist_strategies = []
    predict_therapist_responses = []

    # Regular expressions to extract the required information
    client_emotion_pattern = r"Client's emotion: (.*?)(?:\n|$)"
    therapist_emotion_pattern = r"Therapist's emotion: (.*?)(?:\n|$)"
    therapist_strategy_pattern = r"Therapist's strategy: (.*?)(?:\n|$)"
    therapist_response_pattern = r"Therapist's response: (.*?)(?:\n|$)"

    # Read the JSONL file line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse the JSON object
            data = json.loads(line.strip())

            # Extract label data
            label_text = data.get('label', '')
            label_client_emotion = re.search(client_emotion_pattern, label_text)
            label_therapist_emotion = re.search(therapist_emotion_pattern, label_text)
            label_therapist_strategy = re.search(therapist_strategy_pattern, label_text)
            label_therapist_response = re.search(therapist_response_pattern, label_text)

            # Extract prediction data
            predict_text = data.get('predict', '')
            predict_client_emotion = re.search(client_emotion_pattern, predict_text)
            predict_therapist_emotion = re.search(therapist_emotion_pattern, predict_text)
            predict_therapist_strategy = re.search(therapist_strategy_pattern, predict_text)
            predict_therapist_response = re.search(therapist_response_pattern, predict_text)

            # Append extracted data to respective lists (converted to lowercase)
            label_client_emotions.append(label_client_emotion.group(1).strip().lower() if label_client_emotion else None)
            label_therapist_emotions.append(label_therapist_emotion.group(1).strip().lower() if label_therapist_emotion else None)
            label_therapist_strategies.append(label_therapist_strategy.group(1).strip().lower() if label_therapist_strategy else None)
            label_therapist_responses.append(label_therapist_response.group(1).strip().lower() if label_therapist_response else None)

            predict_client_emotions.append(predict_client_emotion.group(1).strip().lower() if predict_client_emotion else None)
            predict_therapist_emotions.append(predict_therapist_emotion.group(1).strip().lower() if predict_therapist_emotion else None)
            predict_therapist_strategies.append(predict_therapist_strategy.group(1).strip().lower() if predict_therapist_strategy else None)
            predict_therapist_responses.append(predict_therapist_response.group(1).strip().lower() if predict_therapist_response else None)

    # Create a DataFrame
    df = pd.DataFrame({
        'label_client_emotion': label_client_emotions,
        'label_therapist_emotion': label_therapist_emotions,
        'label_therapist_strategy': label_therapist_strategies,
        'label_therapist_response': label_therapist_responses,
        'predict_client_emotion': predict_client_emotions,
        'predict_therapist_emotion': predict_therapist_emotions,
        'predict_therapist_strategy': predict_therapist_strategies,
        'predict_therapist_response': predict_therapist_responses
    })

    return df


def evaluate_predictions(df):
    """
    Evaluate classification metrics for emotions and strategies
    """
    results = {}
    
    # Extract true and predicted values for each category
    client_emotion_true = df['label_client_emotion'].tolist()
    client_emotion_pred = df['predict_client_emotion'].tolist()

    therapist_emotion_true = df['label_therapist_emotion'].tolist()
    therapist_emotion_pred = df['predict_therapist_emotion'].tolist()

    therapist_strategy_true = df['label_therapist_strategy'].tolist()
    therapist_strategy_pred = df['predict_therapist_strategy'].tolist()

    # Function to compute metrics
    def calculate_metrics(true_labels, pred_labels, component_name):
        component_results = {}
        
        # Filter out any None values that might exist in the data
        valid_indices = [i for i, (t, p) in enumerate(zip(true_labels, pred_labels))
                         if t is not None and p is not None]

        filtered_true = [true_labels[i] for i in valid_indices]
        filtered_pred = [pred_labels[i] for i in valid_indices]

        if len(filtered_true) == 0:
            print(f"\nNo valid data points for {component_name}")
            component_results["valid_count"] = 0
            return component_results

        component_results["valid_count"] = len(filtered_true)
        
        accuracy = accuracy_score(filtered_true, filtered_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            filtered_true, filtered_pred, average="weighted", zero_division=0
        )

        component_results["accuracy"] = accuracy
        component_results["precision"] = precision
        component_results["recall"] = recall
        component_results["f1"] = f1

        # Count the number of examples per class for this component
        class_counts = defaultdict(int)
        for label in filtered_true:
            class_counts[label] += 1
            
        component_results["class_distribution"] = {
            cls: {"count": count, "percentage": count/len(filtered_true)*100} 
            for cls, count in sorted(class_counts.items())
        }

        # Find error examples
        errors = [(i, t, p) for i, (t, p) in enumerate(zip(true_labels, pred_labels))
                  if t is not None and p is not None and t != p]
        
        component_results["error_examples"] = [
            {"index": i, "true": t, "predicted": p} for i, t, p in errors[:5]
        ]
        
        return component_results

    # Evaluate each component
    results["client_emotion"] = calculate_metrics(client_emotion_true, client_emotion_pred, "Client Emotion")
    results["therapist_emotion"] = calculate_metrics(therapist_emotion_true, therapist_emotion_pred, "Therapist Emotion")
    results["therapist_strategy"] = calculate_metrics(therapist_strategy_true, therapist_strategy_pred, "Therapist Strategy")
    
    return results


def evaluate_text_generation(df):
    """
    Evaluate text generation metrics for therapist responses
    """
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    results = {}
    
    # Extract ground truth and predicted therapist responses
    references = df['label_therapist_response'].tolist()
    hypotheses = df['predict_therapist_response'].tolist()

    # Filter out None values
    valid_pairs = [(ref, hyp) for ref, hyp in zip(references, hypotheses)
                  if pd.notna(ref) and pd.notna(hyp)]

    if not valid_pairs:
        print("No valid pairs of therapist responses found for evaluation.")
        results["valid_count"] = 0
        return results

    references = [pair[0] for pair in valid_pairs]
    hypotheses = [pair[1] for pair in valid_pairs]
    
    results["valid_count"] = len(references)

    # Tokenize references and hypotheses for BLEU and METEOR
    ref_tokenized = [word_tokenize(str(ref).lower()) for ref in references]
    hyp_tokenized = [word_tokenize(str(hyp).lower()) for hyp in hypotheses]

    # 1. BLEU-2
    bleu2_scores = []
    for ref, hyp in zip(ref_tokenized, hyp_tokenized):
        if ref and hyp:  # Ensure neither is empty
            bleu2_scores.append(sentence_bleu([ref], hyp, weights=(0.99, 0.01)))
        else:
            bleu2_scores.append(0)
    avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0
    results["bleu2"] = avg_bleu2

    # 2. ROUGE (1, 2, L)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores = [], []
    rougeL_f_scores, rougeL_precision_scores, rougeL_recall_scores = [], [], []

    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(str(ref), str(hyp))
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        
        # Extract all three ROUGE-L metrics
        rougeL_f_scores.append(scores['rougeL'].fmeasure)
        rougeL_precision_scores.append(scores['rougeL'].precision)
        rougeL_recall_scores.append(scores['rougeL'].recall)

        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
        avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0

        # Calculate averages for all three ROUGE-L metrics
    avg_rougeL_f = sum(rougeL_f_scores) / len(rougeL_f_scores) if rougeL_f_scores else 0
    avg_rougeL_precision = sum(rougeL_precision_scores) / len(rougeL_precision_scores) if rougeL_precision_scores else 0
    avg_rougeL_recall = sum(rougeL_recall_scores) / len(rougeL_recall_scores) if rougeL_recall_scores else 0

    results["rouge1"] = avg_rouge1
    results["rouge2"] = avg_rouge2
    results["rougeL_f1"] = avg_rougeL_f
    results["rougeL_precision"] = avg_rougeL_precision
    results["rougeL_recall"] = avg_rougeL_recall

    # 3. METEOR
    meteor_scores = []
    for ref, hyp in zip(ref_tokenized, hyp_tokenized):
        if ref and hyp:  # Ensure neither is empty
            meteor_scores.append(meteor_score([ref], hyp))
        else:
            meteor_scores.append(0)
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    results["meteor"] = avg_meteor

    # 4. BERTScore
    # Convert lists to string format for BERTScore
    str_references = [str(ref) for ref in references]
    str_hypotheses = [str(hyp) for hyp in hypotheses]

    P, R, F1 = bert_score(str_hypotheses, str_references, lang="en", verbose=False)
    avg_bertscore_f1 = F1.mean().item()
    results["bertscore_f1"] = avg_bertscore_f1

    # Sample pairs for examples
    sample_size = min(5, len(references))
    indices = np.random.choice(len(references), sample_size, replace=False)
    
    results["examples"] = []
    for i, idx in enumerate(indices):
        results["examples"].append({
            "reference": references[idx],
            "hypothesis": hypotheses[idx],
            "bleu2": bleu2_scores[idx],
            "rougeL_precision": rougeL_precision_scores[idx],
            "rougeL_recall": rougeL_recall_scores[idx],
            "rougeL_f1": rougeL_f_scores[idx]
        })
    
    return results


def print_classification_results(results):
    """Print classification evaluation results in a readable format"""
    for component, metrics in [
        ("Client Emotion", results["client_emotion"]),
        ("Therapist Emotion", results["therapist_emotion"]),
        ("Therapist Strategy", results["therapist_strategy"])
    ]:
        print(f"\nMetrics for {component}:")
        if metrics["valid_count"] == 0:
            print("  No valid data points for evaluation")
            continue
            
        print(f"  Valid examples: {metrics['valid_count']}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision (weighted): {metrics['precision']:.4f}")
        print(f"  Recall (weighted): {metrics['recall']:.4f}")
        print(f"  F1-Score (weighted): {metrics['f1']:.4f}")
        
        print(f"\n  Class distribution for {component}:")
        for cls, info in metrics["class_distribution"].items():
            print(f"    {cls}: {info['count']} examples ({info['percentage']:.1f}%)")
        
        print(f"\n  Example errors for {component}:")
        if not metrics["error_examples"]:
            print("    No errors found")
        else:
            for example in metrics["error_examples"]:
                print(f"    Example {example['index']}: True: '{example['true']}' | Predicted: '{example['predicted']}'")


def print_text_generation_results(results):
    """Print text generation evaluation results in a readable format"""
    print(f"\nEvaluation Metrics for Therapist Response:")
    if results["valid_count"] == 0:
        print("  No valid pairs of therapist responses found for evaluation.")
        return
        
    print(f"  Evaluating {results['valid_count']} pairs of therapist responses")
    print(f"  BLEU-2: {results['bleu2']:.4f}")
    print(f"  ROUGE-1: {results['rouge1']:.4f}")
    print(f"  ROUGE-2: {results['rouge2']:.4f}")
    print(f"  ROUGE-L-F1: {results['rougeL_f1']:.4f}")
    print(f"  ROUGE-L-P: {results['rougeL_precision']:.4f}")
    print(f"  ROUGE-L-R: {results['rougeL_recall']:.4f}")
    print(f"  METEOR: {results['meteor']:.4f}")
    print(f"  BERTScore (F1): {results['bertscore_f1']:.4f}")
    
    print("\nExample Pairs (Ground Truth vs Prediction):")
    for i, example in enumerate(results["examples"]):
        print(f"\nExample {i+1}:")
        print(f"  Reference: {example['reference']}")
        print(f"  Hypothesis: {example['hypothesis']}")
        print(f"  BLEU-2: {example['bleu2']:.4f}, ROUGE-L: {example['rougeL_f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate therapy conversation predictions")
    parser.add_argument("input", help="Path to input JSONL file")
    parser.add_argument("--no-gen-eval", action="store_true", help="Skip text generation evaluation")
    parser.add_argument("--no-cls-eval", action="store_true", help="Skip classification evaluation")
    parser.add_argument("--json-output", "-o", help="Path to save results as JSON (optional)")
    
    args = parser.parse_args()
    
    print(f"Processing JSONL file: {args.input}")
    df = process_jsonl(args.input)
    
    results = {"classification": None, "generation": None}
    
    # Run evaluations
    if not args.no_cls_eval:
        print("\n----- Classification Evaluation -----")
        cls_results = evaluate_predictions(df)
        print_classification_results(cls_results)
        results["classification"] = cls_results
        
    if not args.no_gen_eval:
        print("\n----- Text Generation Evaluation -----")
        gen_results = evaluate_text_generation(df)
        print_text_generation_results(gen_results)
        results["generation"] = gen_results
    
    # Optionally save results as JSON
    if args.json_output:
        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to: {args.json_output}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()