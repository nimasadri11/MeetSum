#from nlg_metrics.rouge.rouge import RougeScorer as ROUGE
#from nlg_metrics.bertscore.bert_util import bert_cos_score_idf, get_model, model2layers, cache_scibert
from nlg_metrics.factscore.fact_extractor import AllenNLPFactExtractor, KnowItAllFactExtractor
from nlg_metrics.factscore.sent_encoder import InferSentSentenceEncoder, GoogleUniversalSentenceEncoder
from nlg_metrics.factscore.fact_connector import BasicFactConnector
from nlg_metrics.factscore.score_calculator import DotProductScoreCalculator, DotProductWithThresholdScoreCalculator
from nlg_metrics.factscore.plot_util import plot_heatmap
from collections import defaultdict
from typing import List
import re
import numpy as np
import json
import torch
import sys
import os

class Scorer(object):
    '''
    # Usage:
    from scorer import Scorer
    scorer = Scorer()
    score = scorer.score(gens, refs)
    # score ranges from 0 to 100
    '''
    def __init__(self, name: str):
        self.name = name

    def score(self, gens: List[List[str]], refs: List[List[str]]) -> List[float]:
        raise NotImplementedError()


class FactScorer(Scorer):
    def __init__(self, tags=['<t>', '</t>'], encoder_type='GoogleUniversalSentenceEncoder', extractor_type='KnowItAllFactExtractor', scorer_type='DotProductWithThresholdScoreCalculator', threshold=0):
        super(FactScorer, self).__init__(name='FactScorer')
        self.encoder_type = encoder_type
        self.extractor_type = extractor_type
        self.scorer_type = scorer_type
        self.threshold = threshold
        self.tags = tags
        
    @property
    def extractor(self):
        if not hasattr(self, '_extractor'):
            if self.extractor_type == 'KnowItAllFactExtractor':
                self._extractor = KnowItAllFactExtractor()
            elif self.extractor_type == 'AllenNLPFactExtractor': 
                self._extractor = AllenNLPFactExtractor()
        return self._extractor

    @property
    def encoder(self):
        if not hasattr(self, '_encoder'):
            if self.encoder_type == 'GoogleUniversalSentenceEncoder':
                self._encoder = GoogleUniversalSentenceEncoder()
            elif self.encoder_type == 'InferSentSentenceEncoder':
                self._encoder = InferSentSentenceEncoder()
        return self._encoder

    @property
    def connector(self):
        if not hasattr(self, '_connector'):
            self._connector = BasicFactConnector()
        return self._connector

    @property
    def calculator(self):
        if not hasattr(self, '_calculator'):
            if self.scorer_type == 'DotProductWithThresholdScoreCalculator':
                self._calculator = DotProductWithThresholdScoreCalculator(self.threshold)
            else:
                self._calculator = DotProductScoreCalculator()
        return self._calculator
        
    def score(self, gens: List[str], refs: List[str], return_all=False) -> List[float]:
        print("in score")
        gens_facts = self.extractor.extract_list(gens)
        print(1)
        refs_facts = self.extractor.extract_list(refs)
        print(2)
        gens_sents = [self.connector.connect_list(facts) for facts in gens_facts]
        print(3)
        refs_sents = [self.connector.connect_list(facts) for facts in refs_facts]
        print(4)
        gens_embs = [self.encoder.encode_list(sents) for sents in gens_sents]
        print(5)
        refs_embs = [self.encoder.encode_list(sents) for sents in refs_sents]
        print(6)
        scores = self.calculator.calculate_list(gens_embs, refs_embs)
        print(7)
        if return_all:
            print("returnall")
            return (gens_facts, refs_facts, gens_sents, refs_sents, gens_embs, refs_embs, scores)
        print('return')
        return [100 * score['f1'] for score in scores]

    def score_each(self, gen: str, ref: str, return_all=False, verbose=False, heatmap=False) -> float:
        assert(type(gen) is str and type(ref) is str)
        gen_facts, ref_facts, gen_sents, ref_sents, gen_embs, ref_embs, score = self.score([gen], [ref], return_all=True)
        gen_facts, ref_facts, gen_sents, ref_sents, gen_embs, ref_embs, score = gen_facts[0], ref_facts[0], gen_sents[0], ref_sents[0], gen_embs[0], ref_embs[0], score[0]
        if verbose:
            print('Generated Facts:')
            self.pretty_print(gen_facts, prefix='GEN')
            print('Reference Facts:')
            self.pretty_print(ref_facts, prefix='REF')
        if heatmap:
            mat = score['mat']
            if mat is not None:
                v, h = mat.shape
                v_names, h_names = [f'G{i}' for i in range(v)], [f'R{i}' for i in range(h)]
                plot_heatmap(h_names, v_names, mat * 100)
        if return_all:
            return gen_facts, ref_facts, gen_sents, ref_sents, gen_embs, ref_embs, score
        return 100 * score['f1']

    def pretty_print(self, facts: List[List[str]], prefix='R'):
        print('-' * 63)
        for i, fact in enumerate(facts):
            print(f'{prefix}{i}: {fact}')
        print('-' * 63)



if __name__ == '__main__':
    def aeq(src, tgt, epsilon=1e-3):
        assert(type(src) == type(tgt)), f'type {type(src)} != type {type(tgt)}'
        if type(src) is list: assert(all(-epsilon < i - j < epsilon for i, j in zip(src, tgt))), f'{src} != {tgt}'
        else: assert(-epsilon < src - tgt < epsilon), f'{src} != {tgt}'

    # Test Case #1 for FACT
    scorer = FactScorer()
    global1 = ""
    global2 = ""
    for file in os.listdir('./meetingOnly/decoded'):
        sub = file[:file.find('_')]
        with open('./meetingOnly/decoded/' + sub + '_decoded.txt', 'r') as f:
            global1 = f.read().replace('\n', '')
        with open('./meetingOnly/reference/' + sub + '_reference.txt', 'r') as f2:
            global2 = f2.read().replace('\n', '')

        print("here1")
        print("g1")
        print(global1)
        print("g2")
        print(global2)
        scores = scorer.score([global1], [global2])
        print("ERROR:", scores)

