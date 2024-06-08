# Evaluate on GenAI-Bench-Image (with 527 prompt) using a specific model
# Example scripts to run:
# VQAScore: python genai_image_eval.py --model clip-flant5-xxl
# CLIPScore: python genai_image_eval.py --model openai:ViT-L-14-336
import argparse
import os
import t2v_metrics
from dataset import GenAIBench_Image, calc_metric, calc_pearson
import json
import torch
import numpy as np
import sys


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./datasets", type=str,
                        help='Root directory for saving datasets.')
    parser.add_argument("--cache_dir", default=t2v_metrics.constants.HF_CACHE_DIR, type=str) 
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--model", default="clip-flant5-xl", type=str)
    # parser.add_argument("--model", default="llava-phi-3-mini", type=str)
    # parser.add_argument("--model", default="llava-v1.5-7b", type=str)
    # parser.add_argument("--model", default="bunny-v1.1-4b", type=str)
    # parser.add_argument("--model", default="paligemma-3b-mix-448", type=str)
    parser.add_argument("--question", default=None, type=str)
    parser.add_argument("--answer", default=None, type=str)
    parser.add_argument("--result_dir", default="./genai_image_results", type=str)
    return parser.parse_args()


tag_groups = {
    'basic': ['attribute', 'scene', 'spatial relation', 'action relation', 'part relation', 'basic'],
    'advanced': ['counting', 'comparison', 'differentiation', 'negation', 'universal', 'advanced'],
    'overall': ['basic', 'advanced'] #, 'all']
}

# def show_performance_per_skill(our_scores, dataset, items_name='images', prompt_to_items_name='prompt_to_images', print_std=False, tag_groups=tag_groups):
#     tag_result = {}
#     tag_file = f"{dataset.root_dir}/genai_skills.json"
#     tags = json.load(open(tag_file))
#     items = getattr(dataset, items_name)
#     prompt_to_items = getattr(dataset, prompt_to_items_name)
#     human_scores = [np.array(items[idx]['human_alignment']).mean() for idx in range(len(items))]
#     items_by_model_tag = {}
#     for tag in tags:
#         items_by_model_tag[tag] = {}
#         for prompt_idx in tags[tag]:
#             for image_idx in prompt_to_items[f"{prompt_idx:05d}"]:
#                 model = items[image_idx]['model']
#                 if model not in items_by_model_tag[tag]:
#                     items_by_model_tag[tag][model] = []
#                 items_by_model_tag[tag][model].append(image_idx)
    
#     for tag in tags:
#         # print(f"Tag: {tag}")
#         tag_result[tag] = {}
#         for model in items_by_model_tag[tag]:
#             our_scores_mean = our_scores[items_by_model_tag[tag][model]].mean()
#             our_scores_std = our_scores[items_by_model_tag[tag][model]].std()
#             # print(f"{model} (Metric Score): {our_scores_mean:.2f} +- {our_scores_std:.2f}")
#             human_scores_mean = np.array(human_scores)[items_by_model_tag[tag][model]].mean()
#             human_scores_std = np.array(human_scores)[items_by_model_tag[tag][model]].std()
#             # print(f"{model} (Human Score): {human_scores_mean:.1f} +- {human_scores_std:.1f}")
#             tag_result[tag][model] = {
#                 'metric': {'mean': our_scores_mean, 'std': our_scores_std},
#                 'human': {'mean': human_scores_mean, 'std': human_scores_std},
#             }
#         # print()
        
#     # print("All")
#     tag_result['all'] = {}
#     all_models = items_by_model_tag[tag]
#     for model in all_models:
#         all_model_indices = set()
#         for tag in items_by_model_tag:
#             all_model_indices = all_model_indices.union(set(items_by_model_tag[tag][model]))
#         all_model_indices = list(all_model_indices)
#         our_scores_mean = our_scores[all_model_indices].mean()
#         our_scores_std = our_scores[all_model_indices].std()
#         # print(f"{model} (Metric Score): {our_scores_mean:.2f} +- {our_scores_std:.2f}")
#         human_scores_mean = np.array(human_scores)[all_model_indices].mean()
#         human_scores_std = np.array(human_scores)[all_model_indices].std()
#         # print(f"{model} (Human Score): {human_scores_mean:.1f} +- {human_scores_std:.1f}")
#         tag_result['all'][model] = {
#             'metric': {'mean': our_scores_mean, 'std': our_scores_std},
#             'human': {'mean': human_scores_mean, 'std': human_scores_std},
#         }
    
#     for tag_group in tag_groups:
#         for score_name in ['metric', 'human']:
#             print(f"Tag Group: {tag_group} ({score_name} performance)")
#             tag_header = f"{'Model':<20}" + " ".join([f"{tag:<20}" for tag in tag_groups[tag_group]])
#             print(tag_header)
#             for model_name in all_models:
#                 if print_std:
#                     detailed_scores = [f"{tag_result[tag][model_name][score_name]['mean']:.2f} +- {tag_result[tag][model_name][score_name]['std']:.2f}" for tag in tag_groups[tag_group]]
#                 else:
#                     detailed_scores = [f"{tag_result[tag][model_name][score_name]['mean']:.2f}" for tag in tag_groups[tag_group]]
#                 detailed_scores = " ".join([f"{score:<20}" for score in detailed_scores])
#                 model_scores = f"{model_name:<20}" + detailed_scores
#                 print(model_scores)
#             print()
#         print()

def show_performance_per_skill(our_scores, dataset_obj, items_name='images', prompt_to_items_name='prompt_to_images', print_std=False, tag_groups=tag_groups):
    tag_result = {}
    tag_file = f"{dataset_obj.root_dir}/genai_skills.json"
    tags = json.load(open(tag_file))
    items = getattr(dataset_obj, items_name)
    prompt_to_items = getattr(dataset_obj, prompt_to_items_name)
    human_scores = [np.array(items[idx]['human_alignment']).mean() for idx in range(len(items))]
    items_by_model_tag = {}
    
    for tag in tags:
        items_by_model_tag[tag] = {}
        for prompt_idx in tags[tag]:
            for image_idx in prompt_to_items[f"{prompt_idx:05d}"]:
                model = items[image_idx]['model']
                if model not in items_by_model_tag[tag]:
                    items_by_model_tag[tag][model] = []
                items_by_model_tag[tag][model].append(image_idx)
    
    for tag in tags:
        tag_result[tag] = {}
        mscore, hscore = [], []
        for model in items_by_model_tag[tag]:
            model_indices = items_by_model_tag[tag][model]
            our_scores_subset = our_scores[model_indices].flatten()
            human_scores_subset = np.array(human_scores)[model_indices]
            pearson_corr = calc_pearson(our_scores_subset, human_scores_subset)
            kendall_tau = calc_metric(human_scores_subset, our_scores_subset, variant="tau_b")
            pairwise, _ = calc_metric(human_scores_subset, our_scores_subset, variant="pairwise_acc_with_tie_optimization")
            
            tag_result[tag][model] = {
                'pearson': pearson_corr,
                'kendall_tau': kendall_tau, 
                'pairwise': pairwise,
            }
            mscore.append(our_scores_subset)
            hscore.append(human_scores_subset)

        mscore = np.concatenate(mscore)
        hscore = np.concatenate(hscore)
        pearson_corr = calc_pearson(mscore, hscore)
        kendall_tau = calc_metric(hscore, mscore, variant="tau_b")
        pairwise, _ = calc_metric(hscore, mscore, variant="pairwise_acc_with_tie_optimization")
        tag_result[tag]['all'] = {
            'pearson': pearson_corr,
            'kendall_tau': kendall_tau, 
            'pairwise': pairwise,
        }

    # Print results for each tag group
    for tag_group in tag_groups:
        print(f"Tag Group: {tag_group} (Performance Metrics)")
        tag_header = f"{'Model':<20}" + " ".join([f"{tag:<32}" for tag in tag_groups[tag_group]])
        print(tag_header)
        for model_name in set(model for tag in tag_groups[tag_group] for model in items_by_model_tag[tag]):
            scores_line = f"{model_name:<20}"
            for tag in tag_groups[tag_group]:
                scores = tag_result[tag].get(model_name, {'pearson': 0, 'kendall_tau': 0, 'pairwise': 0})
                scores_line += f"P: {scores['pearson']:.2f} | K: {scores['kendall_tau']:.2f} | PW: {scores['pairwise']:.2f}    "
            print(scores_line)
        # Print "All" row
        all_scores_line = f"{'All':<20}"
        for tag in tag_groups[tag_group]:
            all_scores = tag_result[tag]['all']
            all_scores_line += f"P: {all_scores['pearson']:.2f} | K: {all_scores['kendall_tau']:.2f} | PW: {all_scores['pairwise']:.2f}    "
        print(all_scores_line)
        print() 

def main():
    args = config()
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)
    
    os.makedirs(args.result_dir, exist_ok=True)
    dataset = GenAIBench_Image(root_dir=args.root_dir)
    result_path = f"{args.result_dir}/{args.model}_527_prompts.pt"
    if os.path.exists(result_path):
        print(f"Result file {result_path} already exists. Skipping.")
        scores = torch.load(result_path)
    else:
        score_func = t2v_metrics.get_score_model(model=args.model, device=args.device, cache_dir=args.cache_dir)

        kwargs = {}
        if args.question is not None:
            print(f"Using question template: {args.question}")
            kwargs['question_template'] = args.question
        if args.answer is not None:
            print(f"Using answer template: {args.answer}")
            kwargs['answer_template'] = args.answer
        
        print(f"Performance of {args.model}.")
        scores = score_func.batch_forward(dataset, batch_size=args.batch_size, **kwargs).cpu()
        torch.save(scores, result_path)
    
    original_stdout = sys.stdout
    with open(f"{args.result_dir}/{args.model}_metrics.txt", 'w') as f:
        sys.stdout = f
        ### Get performance per skill
        our_scores = scores.mean(axis=1)
        show_performance_per_skill(our_scores, dataset, print_std=True)    

        print("Alignment Performance")
        ### Alignment performance
        dataset.evaluate_scores(scores)

        sys.stdout = original_stdout
    

if __name__ == "__main__":
    main()
