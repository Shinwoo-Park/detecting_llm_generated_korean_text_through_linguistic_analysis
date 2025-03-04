import json 
import pickle 
import numpy as np
import random 
import argparse

# 입력: 글 1개에 대한 문장, 형태소, 품사 정보
# Input: Sentence, Morpheme, and Part-of-Speech Information for a Single Text
def analyze_comma_usage(sentences, morphs, pos):

    results = {}

    # 글 1건에서 사용된 총 쉼표 개수 
    # Total Number of Commas Used in a Single Text
    total_comma_count_per_text = 0

    # 글 1건을 구성하는 문장들에서 사용된 쉼표 개수의 평균
    # Average Number of Commas Used per Sentence in a Single Text
    num_commas_per_sentence = []

    # 글 1건에서 쉼표를 포함하고 있는 문장의 비율
    # Proportion of Sentences Containing Commas in a Single Text
    num_sentences = len(sentences)
    num_comma_include_sentences = 0

    # 글 1건을 구성하는 문장들에서 쉼표가 사용된 위치의 상대적 위치
    # Relative Position of Commas Used in Sentences in a Single Text
    relative_positions_per_sentence = []
    
    # 글 1건을 구성하는 문장들에서 쉼표로 나뉜 분절 길이
    # Length of Segments Divided by Commas in Sentences in a Single Text
    segment_lengths_per_sentence = []
    
    # 글 1건을 구성하는 문장들에서 쉼표 앞뒤 형태소 품사
    # Part-of-Speech of Morphemes Before and After Commas in Sentences in a Single Text
    pos_patterns_per_sentence = []

    # 글 1건을 구성하는 문장들에서 쉼표 앞뒤 형태소 품사 다양성 점수
    # Diversity Score of Part-of-Speech of Morphemes Before and After Commas in Sentences in a Single Text
    pos_patterns_diversity_score_per_sentence = []
    
    # 글 1건을 구성하는 문장들에서 쉼표 사용 비율
    # Rate of Comma Usage in Sentences in a Single Text
    comma_usage_rate_per_sentence = []
    
    # 글 1건에 대하여 쉼표가 사용된 위치의 평균값 및 표준편차
    # Average and Standard Deviation of Relative Position of Commas Used in a Single Text
    avg_relative_position_per_sentence = []
    std_relative_position_per_sentence = []
    
    # 글 1건에 대하여 쉼표로 나뉜 문장 분절 길이의 평균값 및 표준편차
    # Average and Standard Deviation of Length of Segments Divided by Commas in a Single Text
    avg_segment_length_per_sentence = []
    std_segment_length_per_sentence = []

    for morp, p in zip(morphs, pos):

        if ',' in morp:
            num_comma_include_sentences += 1

        commas = [i for i, m in enumerate(morp) if m == ',']
        num_commas = len(commas) 
        total_comma_count_per_text += num_commas
        num_commas_per_sentence.append(num_commas)

        if num_commas > 0: 
            relative_positions = [comma / len(morp) for comma in commas]
            relative_positions_per_sentence.append(relative_positions)
            # 쉼표로 잘린 분절 길이 (쉼표는 길이 계산에서 제외)
            # Length of segments divided by commas (excluding commas in length calculation)
            segment_lengths = [len(morp[start+1:end]) if start in commas else len(morp[start:end]) 
                            for start, end in zip([0] + commas, commas + [len(morp)])]
            segment_lengths_per_sentence.append(segment_lengths)
            # 쉼표 앞뒤 형태소 품사
            # Part-of-speech of morphemes before and after commas
            pos_patterns = [(p[i-1], p[i+1]) for i in commas if 0 < i < len(p)-1] 
            pos_patterns_per_sentence.append(pos_patterns)

            # 쉼표 앞뒤 형태소 품사 다양성 점수
            # Diversity score of part-of-speech of morphemes before and after commas
            if len(pos_patterns) == 0:
                pos_patterns_diversity_score = 0
            else:
                pos_patterns_diversity_score = len(set(pos_patterns)) / len(pos_patterns)   
            pos_patterns_diversity_score_per_sentence.append(pos_patterns_diversity_score)

            num_morphs = len(morp)
            comma_usage_rate_per_sentence.append(num_commas / num_morphs)

            avg_relative_position = np.mean(relative_positions)
            std_relative_position = np.std(relative_positions)
            avg_segment_length = np.mean(segment_lengths)
            std_segment_length = np.std(segment_lengths)

            avg_relative_position_per_sentence.append(avg_relative_position)
            std_relative_position_per_sentence.append(std_relative_position)
            avg_segment_length_per_sentence.append(avg_segment_length)
            std_segment_length_per_sentence.append(std_segment_length)

        else: 
            relative_positions_per_sentence.append([])
            segment_lengths_per_sentence.append([])
            pos_patterns_per_sentence.append([])
            comma_usage_rate_per_sentence.append(0)

            # 쉼표 앞뒤 형태소 품사 다양성 점수
            # Diversity score of part-of-speech of morphemes before and after commas
            pos_patterns_diversity_score_per_sentence.append(0)

            avg_relative_position_per_sentence.append(0)
            std_relative_position_per_sentence.append(0)
            avg_segment_length_per_sentence.append(0)
            std_segment_length_per_sentence.append(0)

    # 글 1건에 대하여 쉼표를 포함하고 있는 문장의 비율 / 쉼표를 포함하고 있는 문장 수를 전체 문장 수로 나눈 값
    # Proportion of Sentences Containing Commas in a Single Text
    comma_include_rate_per_text = num_comma_include_sentences / num_sentences
    results['comma_include_sentence_rate_per_text'] = comma_include_rate_per_text

    # 글 1건에 대하여 쉼표를 포함하고 있는 문장의 수
    # Number of Sentences Containing Commas in a Single Text
    results['num_comma_include_sentences_per_text'] = num_comma_include_sentences

    # 글 1건에서 사용된 총 쉼표 개수 
    # Total Number of Commas Used in a Single Text
    results['total_comma_count_per_text'] = total_comma_count_per_text

    # 글 1건을 구성하는 문장들에서 전체 형태소 개수 대비 쉼표 개수의 비율의 평균값
    # Average Rate of Commas Used per Sentence in a Single Text
    avg_comma_usage_rate_per_text = np.mean(comma_usage_rate_per_sentence)
    results['avg_comma_usage_rate_per_text'] = avg_comma_usage_rate_per_text

    # 글 1건에 대하여 쉼표가 사용된 위치의 평균값
    # Average Relative Position of Commas Used in a Single Text
    avg_relative_position_per_text = np.mean(avg_relative_position_per_sentence)
    results['avg_relative_position_per_text'] = avg_relative_position_per_text
    # 글 1건에 대하여 쉼표가 사용된 위치의 표준편차
    # Standard Deviation of Relative Position of Commas Used in a Single Text
    std_relative_position_per_text = np.std(avg_relative_position_per_sentence)
    results['std_relative_position_per_text'] = std_relative_position_per_text

    # 글 1건에 대하여 쉼표로 나뉜 문장 분절 길이의 평균값
    # Average Length of Segments Divided by Commas in a Single Text
    avg_segment_length_per_text = np.mean(avg_segment_length_per_sentence)
    results['avg_segment_length_per_text'] = avg_segment_length_per_text
    # 글 1건에 대하여 쉼표로 나뉜 문장 분절 길이의 표준편차
    # Standard Deviation of Length of Segments Divided by Commas in a Single Text
    std_segment_length_per_text = np.std(avg_segment_length_per_sentence)
    results['std_segment_length_per_text'] = std_segment_length_per_text

    # 글 1건에 대하여 각 문장별 쉼표 앞뒤 형태소 품사 다양성 점수
    # Diversity Score of Part-of-Speech of Morphemes Before and After Commas in Sentences in a Single Text
    avg_pos_patterns_diversity_score_per_text = np.mean(pos_patterns_diversity_score_per_sentence)
    results['avg_pos_patterns_diversity_score_per_text'] = avg_pos_patterns_diversity_score_per_text

    return results

def analyze_pos_ngram_diversity(pos):
    pos_1grams_per_sentence = []
    pos_2grams_per_sentence = []
    pos_3grams_per_sentence = []
    pos_4grams_per_sentence = []
    pos_5grams_per_sentence = []

    pos_1grams_diversity_per_sentence = []
    pos_2grams_diversity_per_sentence = []
    pos_3grams_diversity_per_sentence = []
    pos_4grams_diversity_per_sentence = []
    pos_5grams_diversity_per_sentence = []

    pos_num_1grams_per_text = []
    pos_num_2grams_per_text = []
    pos_num_3grams_per_text = []
    pos_num_4grams_per_text = []
    pos_num_5grams_per_text = []

    for p in pos:
        unigrams = p
        bigrams = [(p[i-1], p[i]) for i in range(1, len(p))]
        trigrams = [(p[i-2], p[i-1], p[i]) for i in range(2, len(p))]
        fourgrams = [(p[i-3], p[i-2], p[i-1], p[i]) for i in range(3, len(p))]
        fivegrams = [(p[i-4], p[i-3], p[i-2], p[i-1], p[i]) for i in range(4, len(p))]

        pos_1grams_per_sentence.append(unigrams)
        pos_2grams_per_sentence.append(bigrams)
        pos_3grams_per_sentence.append(trigrams)
        pos_4grams_per_sentence.append(fourgrams)
        pos_5grams_per_sentence.append(fivegrams)

        pos_num_1grams_per_text.append(len(unigrams))
        pos_num_2grams_per_text.append(len(bigrams))
        pos_num_3grams_per_text.append(len(trigrams))
        pos_num_4grams_per_text.append(len(fourgrams))
        pos_num_5grams_per_text.append(len(fivegrams))

        pos_1grams_diversity_per_sentence.append(len(set(unigrams)) / len(unigrams) if len(unigrams) > 0 else 0)
        pos_2grams_diversity_per_sentence.append(len(set(bigrams)) / len(bigrams) if len(bigrams) > 0 else 0)
        pos_3grams_diversity_per_sentence.append(len(set(trigrams)) / len(trigrams) if len(trigrams) > 0 else 0)
        pos_4grams_diversity_per_sentence.append(len(set(fourgrams)) / len(fourgrams) if len(fourgrams) > 0 else 0)
        pos_5grams_diversity_per_sentence.append(len(set(fivegrams)) / len(fivegrams) if len(fivegrams) > 0 else 0)

    # 글 1건에서 사용된 총 POS 1-gram 개수 / 2-gram 개수 / 3-gram 개수 / 4-gram 개수 / 5-gram 개수
    # Total Number of POS 1-gram / 2-gram / 3-gram / 4-gram / 5-gram Used in a Single Text
    results = {}
    results['pos_num_1grams_per_text'] = sum(pos_num_1grams_per_text)
    results['pos_num_2grams_per_text'] = sum(pos_num_2grams_per_text)
    results['pos_num_3grams_per_text'] = sum(pos_num_3grams_per_text)
    results['pos_num_4grams_per_text'] = sum(pos_num_4grams_per_text)
    results['pos_num_5grams_per_text'] = sum(pos_num_5grams_per_text)

    # 글 1건을 구성하는 문장들에서 사용된 POS 1-gram 개수의 평균값 / 2-gram 개수의 평균값 / 3-gram 개수의 평균값 / 4-gram 개수의 평균값 / 5-gram 개수의 평균값
    # Average Number of POS 1-gram / 2-gram / 3-gram / 4-gram / 5-gram Used per Sentence in a Single Text
    results['avg_pos_num_1grams_per_text'] = np.mean(pos_num_1grams_per_text)
    results['avg_pos_num_2grams_per_text'] = np.mean(pos_num_2grams_per_text)
    results['avg_pos_num_3grams_per_text'] = np.mean(pos_num_3grams_per_text)
    results['avg_pos_num_4grams_per_text'] = np.mean(pos_num_4grams_per_text)
    results['avg_pos_num_5grams_per_text'] = np.mean(pos_num_5grams_per_text)

    # 글 1건을 구성하는 문장들에서 사용된 POS 1-gram 개수의 표준편차 / 2-gram 개수의 표준편차 / 3-gram 개수의 표준편차 / 4-gram 개수의 표준편차 / 5-gram 개수의 표준편차
    # Standard Deviation of Number of POS 1-gram / 2-gram / 3-gram / 4-gram / 5-gram Used per Sentence in a Single Text
    results['std_pos_num_1grams_per_text'] = np.std(pos_num_1grams_per_text)
    results['std_pos_num_2grams_per_text'] = np.std(pos_num_2grams_per_text)
    results['std_pos_num_3grams_per_text'] = np.std(pos_num_3grams_per_text)
    results['std_pos_num_4grams_per_text'] = np.std(pos_num_4grams_per_text)
    results['std_pos_num_5grams_per_text'] = np.std(pos_num_5grams_per_text)

    # 글 1건을 구성하는 문장들에서 POS 1-gram 다양성 점수의 평균값
    # Average POS 1-gram Diversity Score per Sentence in a Single Text
    # POS 1-gram 다양성 점수 = 고유한 POS 1-gram 수 / 전체 POS 1-gram 수
    # POS 1-gram Diversity Score = Number of Unique POS 1-grams / Total Number of POS 1-grams
    avg_pos_1grams_diversity_per_text = np.mean(pos_1grams_diversity_per_sentence)
    results['avg_pos_1grams_diversity_per_text'] = avg_pos_1grams_diversity_per_text
    avg_pos_2grams_diversity_per_text = np.mean(pos_2grams_diversity_per_sentence)
    results['avg_pos_2grams_diversity_per_text'] = avg_pos_2grams_diversity_per_text
    avg_pos_3grams_diversity_per_text = np.mean(pos_3grams_diversity_per_sentence)
    results['avg_pos_3grams_diversity_per_text'] = avg_pos_3grams_diversity_per_text
    avg_pos_4grams_diversity_per_text = np.mean(pos_4grams_diversity_per_sentence)
    results['avg_pos_4grams_diversity_per_text'] = avg_pos_4grams_diversity_per_text
    avg_pos_5grams_diversity_per_text = np.mean(pos_5grams_diversity_per_sentence)
    results['avg_pos_5grams_diversity_per_text'] = avg_pos_5grams_diversity_per_text

    # 글 1건을 구성하는 문장들에서 POS 1-gram 다양성 점수의 표준편차
    # Standard Deviation of POS 1-gram Diversity Score per Sentence in a Single Text
    std_pos_1grams_diversity_per_text = np.std(pos_1grams_diversity_per_sentence)
    results['std_pos_1grams_diversity_per_text'] = std_pos_1grams_diversity_per_text
    std_pos_2grams_diversity_per_text = np.std(pos_2grams_diversity_per_sentence)
    results['std_pos_2grams_diversity_per_text'] = std_pos_2grams_diversity_per_text
    std_pos_3grams_diversity_per_text = np.std(pos_3grams_diversity_per_sentence)
    results['std_pos_3grams_diversity_per_text'] = std_pos_3grams_diversity_per_text
    std_pos_4grams_diversity_per_text = np.std(pos_4grams_diversity_per_sentence)
    results['std_pos_4grams_diversity_per_text'] = std_pos_4grams_diversity_per_text
    std_pos_5grams_diversity_per_text = np.std(pos_5grams_diversity_per_sentence)
    results['std_pos_5grams_diversity_per_text'] = std_pos_5grams_diversity_per_text

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--text_type', type=str, default="abstract") # essay, poetry, abstract
    parser.add_argument('--seed', type=int, default=42) 
    args = parser.parse_args()

    # Load the data
    with open(f"{args.text_type}_pos_taging_results.pkl", "rb") as f:
        sentence_level_ana = pickle.load(f)
    data = []
    with open(f'../katfish_dataset/{args.text_type}.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # Get the data
    human_texts = []
    gpt_texts = []
    solar_texts = []
    qwen_texts = []
    llama_texts = []
    llm_texts = []
    for inst in data: 
        if inst['written_by'] == 'human': 
            human_texts.append(inst['text'])
        else: 
            llm_texts.append(inst['text'])
            if inst['written_by'] == 'gpt-4o-2024-05-13': 
                gpt_texts.append(inst['text'])
            elif inst['written_by'] == 'solar-1-mini-chat': 
                solar_texts.append(inst['text'])
            elif inst['written_by'] == 'qwen2:72b-instruct': 
                qwen_texts.append(inst['text'])
            elif inst['written_by'] == 'llama3.1:70b': 
                llama_texts.append(inst['text'])
    human_id = []
    gpt_id = []
    solar_id = []
    qwen_id = []
    llama_id = []
    id_type = None 
    if args.text_type == 'essay': 
        id_type = 'topic'
    elif args.text_type == 'poetry': 
        id_type = 'poet_number'
    elif args.text_type == 'abstract': 
        id_type = 'title'
    else: 
        pass 
    for inst in data: 
        if inst['written_by'] == 'human': 
            human_id.append(inst[id_type])
        else: 
            if inst['written_by'] == 'gpt-4o-2024-05-13': 
                gpt_id.append(inst[id_type])
            elif inst['written_by'] == 'solar-1-mini-chat':
                solar_id.append(inst[id_type])
            elif inst['written_by'] == 'qwen2:72b-instruct':
                qwen_id.append(inst[id_type])
            elif inst['written_by'] == 'llama3.1:70b':
                llama_id.append(inst[id_type])
            else: 
                print('Error')
    human_sentences = sentence_level_ana['human']['sentences']
    human_morphs = sentence_level_ana['human']['morphs']
    human_pos = sentence_level_ana['human']['pos']
    gpt_sentences = sentence_level_ana['gpt']['sentences']
    gpt_morphs = sentence_level_ana['gpt']['morphs']
    gpt_pos = sentence_level_ana['gpt']['pos']
    solar_sentences = sentence_level_ana['solar']['sentences']
    solar_morphs = sentence_level_ana['solar']['morphs']
    solar_pos = sentence_level_ana['solar']['pos']
    qwen_sentences = sentence_level_ana['qwen']['sentences']
    qwen_morphs = sentence_level_ana['qwen']['morphs']
    qwen_pos = sentence_level_ana['qwen']['pos']
    llama_sentences = sentence_level_ana['llama']['sentences']
    llama_morphs = sentence_level_ana['llama']['morphs']
    llama_pos = sentence_level_ana['llama']['pos']
    llm_sentences = []
    llm_morphs = []
    llm_pos = [] 
    llm_sentences = gpt_sentences + solar_sentences + qwen_sentences + llama_sentences
    llm_morphs = gpt_morphs + solar_morphs + qwen_morphs + llama_morphs
    llm_pos = gpt_pos + solar_pos + qwen_pos + llama_pos

    # Feature Analysis
    human_comma_ana = []
    gpt_comma_ana = []
    solar_comma_ana = []
    qwen_comma_ana = []
    llama_comma_ana = []
    llm_comma_ana = []
    human_pos_ngram_ana = []
    gpt_pos_ngram_ana = []
    solar_pos_ngram_ana = []
    qwen_pos_ngram_ana = []
    llama_pos_ngram_ana = []
    llm_pos_ngram_ana = []
    for human_s, human_m, human_p in zip(human_sentences, human_morphs, human_pos):
        human_comma_ana.append(analyze_comma_usage(human_s, human_m, human_p))
        human_pos_ngram_ana.append(analyze_pos_ngram_diversity(human_p))
    for gpt_s, gpt_m, gpt_p in zip(gpt_sentences, gpt_morphs, gpt_pos):
        gpt_comma_ana.append(analyze_comma_usage(gpt_s, gpt_m, gpt_p))
        gpt_pos_ngram_ana.append(analyze_pos_ngram_diversity(gpt_p))
    for solar_s, solar_m, solar_p in zip(solar_sentences, solar_morphs, solar_pos):
        solar_comma_ana.append(analyze_comma_usage(solar_s, solar_m, solar_p))
        solar_pos_ngram_ana.append(analyze_pos_ngram_diversity(solar_p))
    for qwen_s, qwen_m, qwen_p in zip(qwen_sentences, qwen_morphs, qwen_pos):
        qwen_comma_ana.append(analyze_comma_usage(qwen_s, qwen_m, qwen_p))
        qwen_pos_ngram_ana.append(analyze_pos_ngram_diversity(qwen_p))
    for llama_s, llama_m, llama_p in zip(llama_sentences, llama_morphs, llama_pos):
        llama_comma_ana.append(analyze_comma_usage(llama_s, llama_m, llama_p))
        llama_pos_ngram_ana.append(analyze_pos_ngram_diversity(llama_p))
    for llm_s, llm_m, llm_p in zip(llm_sentences, llm_morphs, llm_pos):
        llm_comma_ana.append(analyze_comma_usage(llm_s, llm_m, llm_p))
        llm_pos_ngram_ana.append(analyze_pos_ngram_diversity(llm_p))

    human_comma_include_sentence_rate_per_text = [ana['comma_include_sentence_rate_per_text'] for ana in human_comma_ana]
    human_num_comma_include_sentences_per_text = [ana['num_comma_include_sentences_per_text'] for ana in human_comma_ana]
    human_total_comma_count_per_text = [ana['total_comma_count_per_text'] for ana in human_comma_ana]
    human_avg_comma_usage_rate_per_text = [ana['avg_comma_usage_rate_per_text'] for ana in human_comma_ana]
    human_avg_relative_position_per_text = [ana['avg_relative_position_per_text'] for ana in human_comma_ana]
    human_std_relative_position_per_text = [ana['std_relative_position_per_text'] for ana in human_comma_ana]
    human_avg_segment_length_per_text = [ana['avg_segment_length_per_text'] for ana in human_comma_ana]
    human_std_segment_length_per_text = [ana['std_segment_length_per_text'] for ana in human_comma_ana]
    human_pos_diversity_score_before_after_comma_per_text = [ana['avg_pos_patterns_diversity_score_per_text'] for ana in human_comma_ana]
    gpt_comma_include_sentence_rate_per_text = [ana['comma_include_sentence_rate_per_text'] for ana in gpt_comma_ana]
    gpt_num_comma_include_sentences_per_text = [ana['num_comma_include_sentences_per_text'] for ana in gpt_comma_ana]
    gpt_total_comma_count_per_text = [ana['total_comma_count_per_text'] for ana in gpt_comma_ana]
    gpt_avg_comma_usage_rate_per_text = [ana['avg_comma_usage_rate_per_text'] for ana in gpt_comma_ana]
    gpt_avg_relative_position_per_text = [ana['avg_relative_position_per_text'] for ana in gpt_comma_ana]
    gpt_std_relative_position_per_text = [ana['std_relative_position_per_text'] for ana in gpt_comma_ana]
    gpt_avg_segment_length_per_text = [ana['avg_segment_length_per_text'] for ana in gpt_comma_ana]
    gpt_std_segment_length_per_text = [ana['std_segment_length_per_text'] for ana in gpt_comma_ana]
    gpt_pos_diversity_score_before_after_comma_per_text = [ana['avg_pos_patterns_diversity_score_per_text'] for ana in gpt_comma_ana]
    solar_comma_include_sentence_rate_per_text = [ana['comma_include_sentence_rate_per_text'] for ana in solar_comma_ana]
    solar_num_comma_include_sentences_per_text = [ana['num_comma_include_sentences_per_text'] for ana in solar_comma_ana]
    solar_total_comma_count_per_text = [ana['total_comma_count_per_text'] for ana in solar_comma_ana]
    solar_avg_comma_usage_rate_per_text = [ana['avg_comma_usage_rate_per_text'] for ana in solar_comma_ana]
    solar_avg_relative_position_per_text = [ana['avg_relative_position_per_text'] for ana in solar_comma_ana]
    solar_std_relative_position_per_text = [ana['std_relative_position_per_text'] for ana in solar_comma_ana]
    solar_avg_segment_length_per_text = [ana['avg_segment_length_per_text'] for ana in solar_comma_ana]
    solar_std_segment_length_per_text = [ana['std_segment_length_per_text'] for ana in solar_comma_ana]
    solar_pos_diversity_score_before_after_comma_per_text = [ana['avg_pos_patterns_diversity_score_per_text'] for ana in solar_comma_ana]
    qwen_comma_include_sentence_rate_per_text = [ana['comma_include_sentence_rate_per_text'] for ana in qwen_comma_ana]
    qwen_num_comma_include_sentences_per_text = [ana['num_comma_include_sentences_per_text'] for ana in qwen_comma_ana]
    qwen_total_comma_count_per_text = [ana['total_comma_count_per_text'] for ana in qwen_comma_ana]
    qwen_avg_comma_usage_rate_per_text = [ana['avg_comma_usage_rate_per_text'] for ana in qwen_comma_ana]
    qwen_avg_relative_position_per_text = [ana['avg_relative_position_per_text'] for ana in qwen_comma_ana]
    qwen_std_relative_position_per_text = [ana['std_relative_position_per_text'] for ana in qwen_comma_ana]
    qwen_avg_segment_length_per_text = [ana['avg_segment_length_per_text'] for ana in qwen_comma_ana]
    qwen_std_segment_length_per_text = [ana['std_segment_length_per_text'] for ana in qwen_comma_ana]
    qwen_pos_diversity_score_before_after_comma_per_text = [ana['avg_pos_patterns_diversity_score_per_text'] for ana in qwen_comma_ana]
    llama_comma_include_sentence_rate_per_text = [ana['comma_include_sentence_rate_per_text'] for ana in llama_comma_ana]
    llama_num_comma_include_sentences_per_text = [ana['num_comma_include_sentences_per_text'] for ana in llama_comma_ana]
    llama_total_comma_count_per_text = [ana['total_comma_count_per_text'] for ana in llama_comma_ana]
    llama_avg_comma_usage_rate_per_text = [ana['avg_comma_usage_rate_per_text'] for ana in llama_comma_ana]
    llama_avg_relative_position_per_text = [ana['avg_relative_position_per_text'] for ana in llama_comma_ana]
    llama_std_relative_position_per_text = [ana['std_relative_position_per_text'] for ana in llama_comma_ana]
    llama_avg_segment_length_per_text = [ana['avg_segment_length_per_text'] for ana in llama_comma_ana]
    llama_std_segment_length_per_text = [ana['std_segment_length_per_text'] for ana in llama_comma_ana]
    llama_pos_diversity_score_before_after_comma_per_text = [ana['avg_pos_patterns_diversity_score_per_text'] for ana in llama_comma_ana]
    llm_comma_include_sentence_rate_per_text = [ana['comma_include_sentence_rate_per_text'] for ana in llm_comma_ana]
    llm_num_comma_include_sentences_per_text = [ana['num_comma_include_sentences_per_text'] for ana in llm_comma_ana]
    llm_total_comma_count_per_text = [ana['total_comma_count_per_text'] for ana in llm_comma_ana]
    llm_avg_comma_usage_rate_per_text = [ana['avg_comma_usage_rate_per_text'] for ana in llm_comma_ana]
    llm_avg_relative_position_per_text = [ana['avg_relative_position_per_text'] for ana in llm_comma_ana]
    llm_std_relative_position_per_text = [ana['std_relative_position_per_text'] for ana in llm_comma_ana]
    llm_avg_segment_length_per_text = [ana['avg_segment_length_per_text'] for ana in llm_comma_ana]
    llm_std_segment_length_per_text = [ana['std_segment_length_per_text'] for ana in llm_comma_ana]
    llm_pos_diversity_score_before_after_comma_per_text = [ana['avg_pos_patterns_diversity_score_per_text'] for ana in llm_comma_ana]

    # Construct data for ML experiment
    human_features = []
    gpt_features = []
    solar_features = []
    qwen_features = []
    llama_features = []
    for include, usage, position, segment, pos_diversity in zip(human_comma_include_sentence_rate_per_text, human_avg_comma_usage_rate_per_text, human_avg_relative_position_per_text, human_avg_segment_length_per_text, human_pos_diversity_score_before_after_comma_per_text):
        human_features.append([include, usage, position, segment, pos_diversity])
    for include, usage, position, segment, pos_diversity in zip(gpt_comma_include_sentence_rate_per_text, gpt_avg_comma_usage_rate_per_text, gpt_avg_relative_position_per_text, gpt_avg_segment_length_per_text, gpt_pos_diversity_score_before_after_comma_per_text):
        gpt_features.append([include, usage, position, segment, pos_diversity])
    for include, usage, position, segment, pos_diversity in zip(solar_comma_include_sentence_rate_per_text, solar_avg_comma_usage_rate_per_text, solar_avg_relative_position_per_text, solar_avg_segment_length_per_text, solar_pos_diversity_score_before_after_comma_per_text):
        solar_features.append([include, usage, position, segment, pos_diversity])
    for include, usage, position, segment, pos_diversity in zip(qwen_comma_include_sentence_rate_per_text, qwen_avg_comma_usage_rate_per_text, qwen_avg_relative_position_per_text, qwen_avg_segment_length_per_text, qwen_pos_diversity_score_before_after_comma_per_text):
        qwen_features.append([include, usage, position, segment, pos_diversity])
    for include, usage, position, segment, pos_diversity in zip(llama_comma_include_sentence_rate_per_text, llama_avg_comma_usage_rate_per_text, llama_avg_relative_position_per_text, llama_avg_segment_length_per_text, llama_pos_diversity_score_before_after_comma_per_text):
        llama_features.append([include, usage, position, segment, pos_diversity])

    human_ml_data = []
    gpt_ml_data = []
    solar_ml_data = []
    qwen_ml_data = []
    llama_ml_data = []
    for text, feature, id in zip(human_texts, human_features, human_id):
        item = {}
        item['text'] = text
        item['feature'] = feature
        item['label'] = 0
        item['written_by'] = 'Human'
        item['id'] = id
        human_ml_data.append(item)
    for text, feature, id in zip(gpt_texts, gpt_features, gpt_id):
        item = {}
        item['text'] = text
        item['feature'] = feature
        item['label'] = 1
        item['written_by'] = 'GPT-4o'
        item['id'] = id
        gpt_ml_data.append(item)
    for text, feature, id in zip(solar_texts, solar_features, solar_id):
        item = {}
        item['text'] = text
        item['feature'] = feature
        item['label'] = 1
        item['written_by'] = 'Solar'
        item['id'] = id
        solar_ml_data.append(item)
    for text, feature, id in zip(qwen_texts, qwen_features, qwen_id):
        item = {}
        item['text'] = text
        item['feature'] = feature
        item['label'] = 1
        item['written_by'] = 'Qwen2'
        item['id'] = id
        qwen_ml_data.append(item)
    for text, feature, id in zip(llama_texts, llama_features, llama_id):
        item = {}
        item['text'] = text
        item['feature'] = feature
        item['label'] = 1
        item['written_by'] = 'Llama3.1'
        item['id'] = id
        llama_ml_data.append(item)

    # Data Split
    # human_ml_data를 seed를 사용하여 8:2로 random 하게 자르기 
    # Randomly divide human_ml_data into 8:2 using seed
    random.seed(args.seed)
    random.shuffle(human_ml_data)
    train_size = int(len(human_ml_data) * 0.8)
    train_human_ml_data = human_ml_data[:train_size]
    test_human_ml_data = human_ml_data[train_size:]

    # Save the data
    ml_data = {} 
    ml_data['train'] = train_human_ml_data + gpt_ml_data 
    ml_data['human_solar_ood'] = test_human_ml_data + solar_ml_data
    ml_data['human_qwen_ood'] = test_human_ml_data + qwen_ml_data
    ml_data['human_llama_ood'] = test_human_ml_data + llama_ml_data

    with open(args.text_type + '_' + str(args.seed) + '_ml_data.pkl', 'wb') as f: 
        pickle.dump(ml_data, f)