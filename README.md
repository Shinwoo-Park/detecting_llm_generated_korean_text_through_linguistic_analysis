# Detecting LLM-Generated Korean Text through Linguistic Feature Analysis

[**📖 Paper**](https://arxiv.org/abs/2503.00032)

## Overview

We introduce the first benchmark dataset **KatFish** for detecting Korean text generated by large language models (LLMs). As LLMs become increasingly capable of producing fluent and coherent text, distinguishing machine-generated content from human-written text is crucial for upholding academic integrity, preventing plagiarism, and maintaining ethical research standards. We also propose a specialized detection method—**KatFishNet**—that exploits linguistic features unique to the Korean language.

## Motivation

The explosive growth of LLMs has blurred the line between human and machine-generated text. While much research has focused on English, Korean presents its own set of challenges:
- **Flexible Word Spacing:** Korean allows significant variability in spacing.
- **Rich Morphological Structure & Complex POS Patterns:** The language’s extensive system of postpositions, verb endings, and inflections produces a wide array of syntactic constructions.
- **Distinctive Punctuation Usage:** Korean punctuation—especially the use of commas—differs from that of English, reflecting subtle stylistic and grammatical preferences.

These characteristics demand a language-specific approach for detecting LLM-generated text. We address this gap by systematically analyzing linguistic features that capture these nuances.

## KatFish Dataset

The KatFish dataset is assembled from Korean texts spanning three major genres:

- **Essay:** Argumentative essays collected from educational corpora, covering diverse topics across elementary, middle, and high school levels. Detailed analyses include topic distributions and statistics segmented by education level.
- **Poetry:** Free-form creative poems that capture the expressive and often unconventional style of Korean poetry. The dataset includes works from various age groups, reflecting differences in stylistic trends.
- **Paper Abstract:** Concise summaries of academic papers characterized by formal language, technical terms, and structured presentation.

For each genre, the dataset comprises:
- **Human-Written Texts:** Sourced from reputable public corpora and manually curated to ensure quality.
- **LLM-Generated Texts:** Produced using a combination of two commercial LLMs (e.g., GPT-4o and Solar) and two open-source LLMs (e.g., Qwen2 and Llama3.1). Each text is generated with carefully designed prompts that mirror the style expected for that genre.

## KatFishNet: The Detection Method

**KatFishNet** is a machine learning–based detector that leverages three primary linguistic feature groups, designed specifically to capture the stylistic discrepancies between human and LLM-generated Korean text:

1. **Word Spacing Patterns**  
   - **Analysis:** Examines the distribution of spaces around bound nouns (BN) and auxiliary predicate elements (VX).  
   - **Metrics:** Computes ratios such as the MMN-BN Space Ratio and BN Space Ratio, which quantify how consistently text adheres to standard spacing rules.

2. **POS N-gram Diversity**  
   - **Analysis:** Measures the diversity of part-of-speech (POS) n-grams, ranging from unigrams up to pentagrams.  
   - **Insight:** Human text typically exhibits a richer variety of syntactic patterns compared to LLM-generated text, which tends to repeat common structures.

3. **Comma Usage Patterns**  
   - **Analysis:** Investigates the frequency and positional characteristics of commas, along with the diversity of POS combinations immediately surrounding them.
   - **Metrics:** Includes measures such as Comma Inclusion Rate, Average Comma Usage Rate, Relative Position of Comma, and POS Diversity before and after commas.
  
By extracting and concatenating these feature vectors, KatFishNet trains lightweight classifiers (using methods like logistic regression, random forests, or support vector machines) to effectively distinguish between human-written and machine-generated text.

## Dataset & Detection Framework

<p align="center">
  <img src="./figure/katfish_overview.png" width="500"/>
</p>

## OOD Evaluation - Unseen LLMs
### Objective
To evaluate the generalization ability of detection methods, we conduct **out-of-distribution (OOD) evaluation** by testing the model’s performance on texts generated by **unseen LLMs**. Since new LLMs frequently emerge with distinct generation patterns, a robust detection model should maintain its accuracy even when faced with previously unseen text sources.

### Methodology
- We split the **human-written text** into an **8:2 ratio**, using **80%** for training and reserving **20%** for evaluation.
- **GPT-4o** is used as the primary training LLM-generated dataset.
- For testing, we evaluate against text generated by **Solar, Qwen2, and Llama3.1**.
- The test dataset is structured into three distinct sets, each containing text from one of the unseen LLMs along with the reserved **20% human-written text**.
- **Zero-shot classification setting**: All detection models are tested on LLM-generated text they have never encountered during training.

## Performance of Detecting LLM-Generated Korean Text
The following table presents the performance of various detection methods across different text genres. 

| Genre          | Detection Methods           | Solar  | Qwen2  | Llama3.1 | Average |
|----------------|-----------------------------|--------|--------|----------|---------|
| **Essay**      | Log-Likelihood              | 83.84  | 23.89  |  66.20   | 57.97   |
|                | Entropy                     | 31.25  | 84.53  |  44.12   |  53.30  |
|                | Log-Rank                    | 78.84  | 20.66  |  61.92   |  53.80  |
|                | LRR                         | 45.08  | 80.56  |  53.15   |  59.59  |
|                | DetectGPT                   | 52.78  | 37.45  |  47.18   |  45.80  |
|                | NPR                         | 55.22  | 19.90  |  44.71   |  39.94  |
|                | LLM Paraphrasing            | 92.08  | 79.74  |  72.00   |  81.27  |
|                | LLM Prompting               | 50.42  | 49.74  |  50.07   |  50.07  |
|                | Fine-tuning                 | 66.77  | 66.65  |  64.37   |  65.93  |
|                | KatFishNet (Word Spacing)   | 86.00  | 80.63  |  71.91   |  79.51  |
|                | KatFishNet (POS Combination) | 92.26  |  83.10  |  73.63   | 82.99 |
|                | KatFishNet (Punctuation)    | 97.57 | 94.63 | 92.45 | **94.88** |
| **Poetry**     | Log-Likelihood              | 77.06 | 47.34 | 59.99 | 61.46 |
|                | Entropy                     | 30.90  | 68.28  | 47.68 | 48.95 |
|                | Log-Rank                    | 75.76  | 45.67  | 60.54 | 60.65 |
|                | LRR                         | 34.40  | 55.86  |  39.79 | 43.35 |
|                | DetectGPT                   | 67.04  | 64.00  |  67.02  | 66.02 |
|                | NPR                         | 63.75  | 41.21  |  62.92 | 55.96 |
|                | LLM Paraphrasing            | 71.32  | 58.79  |  61.51   |  63.87 |
|                | LLM Prompting               | 50.53 | 50.16  | 49.42 | 50.03 |
|                | Fine-tuning                 | 60.35  | 69.61  | 55.96 | 61.97 |
|                | KatFishNet (Word Spacing)   | 71.85  | 65.56  |  43.81   | 60.40   |
|                | KatFishNet (POS Combination) | 39.41  | 79.17  |  53.32   |  57.30  |
|                | KatFishNet (Punctuation)    | 62.65 | 93.45 | 63.22 | **73.10** |
| **Paper Abstract** | Log-Likelihood          | 58.52  |  42.41 |  47.86 |  49.59  |
|                | Entropy                     | 36.13  | 72.64  |  51.85   |  53.54  |
|                | Log-Rank                    | 57.08  | 45.05  |  47.57   |  49.90  |
|                | LRR                         | 49.39  | 47.82  |  54.80   |  50.67  |
|                | DetectGPT                   | 55.81  | 51.70  |  51.11   |  52.87  |
|                | NPR                         | 63.14  | 46.76  |  60.98   | 56.96   |
|                | LLM Paraphrasing            | 70.80  | 36.47  |  64.72   |  57.33  |
|                | LLM Prompting               | 48.60  | 46.41  |  47.18   |  47.39  |
|                | Fine-tuning                 | 50.70  | 49.73  | 50.02    |  50.15  |
|                | KatFishNet (Word Spacing)   | 57.73  | 66.91  |  49.36   |  58.00  |
|                | KatFishNet (POS Combination) | 47.47  | 70.05  |  42.47   |  53.33  |
|                | KatFishNet (Punctuation) | 78.99 | 77.47 | 70.41 | **75.62** |

- Among the baseline methods, **LLM Paraphrasing** achieved the highest performance for essays and abstracts, while **DetectGPT** performed best for poetry.
- **LLM Paraphrasing** outperforms other baselines due to its ability to exploit the statistical patterns in LLM-generated text.
- **KatFishNet with punctuation-based features** significantly improves detection performance, outperforming all other methods in every genre.

## Usage & Implementation

### Environment Setup

Install the necessary dependencies using the provided conda configuration:

```bash
conda env create -f katfish.yaml
conda activate katfish
```

### Running Experiments 

```bash
cd experiment
bash run.sh
```

## Linguistic Analysis

`linguistic_analysis` contains scripts for conducting linguistic analysis on the raw text data:
- **POS Tagging:** The `pos_tagging.py` script performs POS tagging on the original dataset.
- **Comma Feature Analysis:** The `comma_feature_analysis.py` script carries out an in-depth analysis of comma usage patterns.

To execute these analyses, navigate to the `linguistic_analysis` folder and run the following commands in sequence:

```bash
python pos_tagging.py
python comma_feature_analysis.py
```

## 🖊 Citation
```text
@misc{park2025detectingllmgeneratedkoreantext,
      title={Detecting LLM-Generated Korean Text through Linguistic Feature Analysis}, 
      author={Shinwoo Park and Shubin Kim and Do-Kyung Kim and Yo-Sub Han},
      year={2025},
      eprint={2503.00032},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.00032}, 
}
```