from konlpy.tag import Kkma
import kss
from konlpy.utils import pprint
import json 
import pickle 
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--text_type', type=str, default="abstract") # essay, poetry, abstract
    args = parser.parse_args()

    # POS tagger
    kkma = Kkma()

    # Load data
    data = []
    with open(f'../katfish_dataset/{args.text_type}.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line))
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

    # Sentence segmentation
    human_sentences = [] 
    gpt_sentences = [] 
    solar_sentences = [] 
    qwen_sentences = []
    llama_sentences = []
    llm_sentences = []
    for text in human_texts:
        tmp_list = []
        kss_sentences = kss.split_sentences(text)
        for sentence in kss_sentences:
            tmp_list.extend(sentence.split('\n'))
        human_sentences.append(tmp_list)
    for text in gpt_texts:
        tmp_list = []
        kss_sentences = kss.split_sentences(text)
        for sentence in kss_sentences:
            tmp_list.extend(sentence.split('\n'))
        gpt_sentences.append(tmp_list)
    for text in solar_texts:
        tmp_list = []
        kss_sentences = kss.split_sentences(text)
        for sentence in kss_sentences:
            tmp_list.extend(sentence.split('\n'))
        solar_sentences.append(tmp_list)
    for text in qwen_texts:
        tmp_list = []
        kss_sentences = kss.split_sentences(text)
        for sentence in kss_sentences:
            tmp_list.extend(sentence.split('\n'))
        qwen_sentences.append(tmp_list)
    for text in llama_texts:
        tmp_list = []
        kss_sentences = kss.split_sentences(text)
        for sentence in kss_sentences:
            tmp_list.extend(sentence.split('\n'))
        llama_sentences.append(tmp_list)
    for text in llm_texts:
        tmp_list = []
        kss_sentences = kss.split_sentences(text)
        for sentence in kss_sentences:
            tmp_list.extend(sentence.split('\n'))
        llm_sentences.append(tmp_list)

    # POS tagging
    human_sentences_morphs = [] 
    gpt_sentences_morphs = [] 
    solar_sentences_morphs = [] 
    qwen_sentences_morphs = []
    llama_sentences_morphs = []
    llm_sentences_morphs = []
    human_sentences_pos = [] 
    gpt_sentences_pos = [] 
    solar_sentences_pos = [] 
    qwen_sentences_pos = []
    llama_sentences_pos = []
    llm_sentences_pos = []
    for sentences in human_sentences:
        tmp_morph = []
        tmp_pos = []
        for sentence in sentences:
            ana = kkma.pos(sentence)
            morph = []
            pos = [] 
            for item in ana:
                morph.append(item[0])
                pos.append(item[1])
            tmp_morph.append(morph)
            tmp_pos.append(pos)
        human_sentences_morphs.append(tmp_morph)
        human_sentences_pos.append(tmp_pos)
    for sentences in gpt_sentences:
        tmp_morph = []
        tmp_pos = []
        for sentence in sentences:
            ana = kkma.pos(sentence)
            morph = []
            pos = [] 
            for item in ana:
                morph.append(item[0])
                pos.append(item[1])
            tmp_morph.append(morph)
            tmp_pos.append(pos)
        gpt_sentences_morphs.append(tmp_morph)
        gpt_sentences_pos.append(tmp_pos)
    for sentences in solar_sentences:
        tmp_morph = []
        tmp_pos = []
        for sentence in sentences:
            ana = kkma.pos(sentence)
            morph = []
            pos = [] 
            for item in ana:
                morph.append(item[0])
                pos.append(item[1])
            tmp_morph.append(morph)
            tmp_pos.append(pos)
        solar_sentences_morphs.append(tmp_morph)
        solar_sentences_pos.append(tmp_pos)
    for sentences in qwen_sentences:
        tmp_morph = []
        tmp_pos = []
        for sentence in sentences:
            ana = kkma.pos(sentence)
            morph = []
            pos = [] 
            for item in ana:
                morph.append(item[0])
                pos.append(item[1])
            tmp_morph.append(morph)
            tmp_pos.append(pos)
        qwen_sentences_morphs.append(tmp_morph)
        qwen_sentences_pos.append(tmp_pos)
    for sentences in llama_sentences:
        tmp_morph = []
        tmp_pos = []
        for sentence in sentences:
            ana = kkma.pos(sentence)
            morph = []
            pos = [] 
            for item in ana:
                morph.append(item[0])
                pos.append(item[1])
            tmp_morph.append(morph)
            tmp_pos.append(pos)
        llama_sentences_morphs.append(tmp_morph)
        llama_sentences_pos.append(tmp_pos)
    for sentences in llm_sentences:
        tmp_morph = []
        tmp_pos = []
        for sentence in sentences:
            ana = kkma.pos(sentence)
            morph = []
            pos = [] 
            for item in ana:
                morph.append(item[0])
                pos.append(item[1])
            tmp_morph.append(morph)
            tmp_pos.append(pos)
        llm_sentences_morphs.append(tmp_morph)
        llm_sentences_pos.append(tmp_pos)

    # Save data
    total_ana = {} 
    human_ana = {} 
    gpt_ana = {}
    solar_ana = {}
    qwen_ana = {}
    llama_ana = {}
    llm_ana = {}
    human_ana['sentences'] = human_sentences
    human_ana['morphs'] = human_sentences_morphs
    human_ana['pos'] = human_sentences_pos
    gpt_ana['sentences'] = gpt_sentences
    gpt_ana['morphs'] = gpt_sentences_morphs
    gpt_ana['pos'] = gpt_sentences_pos
    solar_ana['sentences'] = solar_sentences
    solar_ana['morphs'] = solar_sentences_morphs
    solar_ana['pos'] = solar_sentences_pos
    qwen_ana['sentences'] = qwen_sentences
    qwen_ana['morphs'] = qwen_sentences_morphs
    qwen_ana['pos'] = qwen_sentences_pos
    llama_ana['sentences'] = llama_sentences
    llama_ana['morphs'] = llama_sentences_morphs
    llama_ana['pos'] = llama_sentences_pos
    llm_ana['sentences'] = llm_sentences
    llm_ana['morphs'] = llm_sentences_morphs
    llm_ana['pos'] = llm_sentences_pos
    total_ana['human'] = human_ana
    total_ana['gpt'] = gpt_ana
    total_ana['solar'] = solar_ana
    total_ana['qwen'] = qwen_ana
    total_ana['llama'] = llama_ana
    total_ana['llm'] = llm_ana

    with open(f'{args.text_type}_pos_taging_results.pkl', 'wb') as f:
        pickle.dump(total_ana, f)