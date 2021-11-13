""" Contains the part of speech tagger class. """
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score
import time
import math
import os
from functools import reduce
import string
START_SYM = '-DOCSTART-'
START_TAG = 'O'


def load_data(sentence_file, tag_file=None):
    """Loads data from two files: one containing sentences and one containing tags.

    tag_file is optional, so this function can be used to load the test data.

    Suggested to split the data by the document-start symbol.

    """
    df = pd.read_csv(sentence_file)
    if tag_file:
        df_y = pd.read_csv(tag_file)
        df['tag']=df_y['tag']
    # return []
    return df

def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions.
    """
    seq = list(data['word'])
    true_tag = list(data['tag'])

    # DEBUG
    start = time.time()
    # seq = seq[:2088] 
    # true_tag = true_tag[:2088]
    
    # seq = ['-DOCSTART-', 'I', 'am', 'Sam','.']
    # true_tag = ['O','PRP','VBP','NNP','.']
    print('evaluate seq len: ',len(seq))
    
    # find unk idx
    unk_idx = []
    for i in range(len(seq)):
        if seq[i] not in model.vocab_set:
            unk_idx.append(i)

    if args.sep_sentence:
        seq = np.array(seq)
        seq_idx = np.where(seq=='.')
        seq_idx = np.concatenate(([0],seq_idx[0]))
        result = []
        for i in range(1,len(seq_idx)):
            # print('seq_idx[i]',seq_idx[i])
            flag=False
            part_seq = seq[seq_idx[i-1]+1:seq_idx[i]+1]
            if part_seq[0]!= '-DOCSTART-':
                part_seq = ['-DOCSTART-'] + list(part_seq)
                flag = True
            res = model.inference(part_seq)
            if flag and i!=1:
                res = res[1:]
            result+=res
    elif args.sep_doc:
        seq = np.array(seq)
        seq_idx = np.where(seq=='-DOCSTART-')
        seq_idx = np.concatenate((seq_idx[0],[len(seq)]))
        result = []
        for i in range(1,len(seq_idx)):
            part_seq = seq[seq_idx[i-1]:seq_idx[i]]
            res = model.inference(part_seq)
            result+=res
    else:
        result = model.inference(seq) 
    unk_res = np.array(result)[unk_idx]
    unk_true_tag = np.array(true_tag)[unk_idx]
    # print('res',res)
    # print('true_tag',true_tag)
    
    acc = accuracy_score(result,true_tag)
    f1 = f1_score(result,true_tag,average='weighted')
    print('Overall\tacc',acc,'\tf1',f1)
    # print('len(unk_res)',len(unk_res))
    # print('len(unk_true_tag)',len(unk_true_tag))
    acc = accuracy_score(unk_res,unk_true_tag)
    f1 = f1_score(unk_res,unk_true_tag,average='weighted')
    print('Unknown\tacc',acc,'\tf1',f1)
    print('seq len:',len(seq),'time used:',time.time()-start)
    return result



class POSTagger():
    def __init__(self,args):
        """Initializes the tagger model parameters and anything else necessary. """
        # self.suf_for_adj = ['ious','sy', 'dy', 'ic', 'ese', 'esque', 'al', 'able', 'ive', 'ish', 'ous', 'zy', 'less', 'ical', 'ly', 'ian', 'ible', 'ful', 'lly', 'i']
        self.suf_for_adj = ['ious','sy', 'dy', 'ic', 'ese', 'esque', 'al', 'able', 'ive', 'ish', 'ous', 'zy', 'ical', 'ly', 'ible', 'ful', 'lly', 'i']
        # self.pre_for_adj = ['un','in']
        self.suf_for_adv = ['ally', 'wards', 'ily', 'wise', 'ward', 'ly']
        self.suf_for_noun = ['ity', 'hood', 'ment', 'al', 'ness', 'acy', 'dom', 'ling', 'ty', 'or', 'ation', 'ship', 'ry', 'ery', 'cy', 'ee', 'age', 'ist', 'ism', 'er', 'action', 'scape', 'ure', 'ion', 'ance', 'ence']
        self.suf_for_verb  = ['ify', 'fy', 'ate', 'en', 'ize', 'ise']
        self.puncs = [p for p in string.punctuation]
        self.unk_list = ['unk_jj', 'unk_adv', 'unk_nn', 'unk_vb','unk_nnp','unk_num','unk_punc']

        self.beam_k = args.beam_k
        self.inference_type = args.inference_type
        self.ngram = args.Ngram
        self.smooth_type = args.smooth_type
        self.rare = args.rare
        self.rare_thr = args.rare_thr
        self.unk_suffix = args.unk_suffix

        if self.smooth_type=='addk':
            self.k = args.addk_param
            # self.k = 0.001 # .5? .05? .01? # bigger, then the trans/emit prob is more far from true val
        self.k = 0.001
        if self.smooth_type=='interpolation':
            if self.ngram == '2':
                self.lambda_coef = [0.8,0.2]
            elif self.ngram == '3':
                # self.lambda_coef = [0.8,0.1,0.1]
                self.lambda_coef = [0.7,0.2,0.1]
            elif self.ngram == '4':
                self.lambda_coef = [0.7,0.15,0.1,0.05]
            self.lambda_coef = np.array(self.lambda_coef)
        pass

    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.
        
        """
        self.vocab, self.tag_trans_cnt, self.emit_cnt, self.tag_num, self.ngram_cnt, self.ngram_prev_cnt,self.interpolation_helper\
                    = self.find_dicts(data) 
        self.tag_set = list(self.tag_num.keys())
        self.tag_len = len(self.tag_set)

        self.vocab_set = list(self.vocab.keys())
        # print(self.vocab_set)
        self.V = len(self.vocab.keys())
        self.reverse_vocab = {self.vocab_set[k]:k for k in range(self.V)}

        self.transition_prob_mat = self.find_transition_prob_mat()
        self.emission_prob_mat = self.find_emission_prob_mat()
        

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        return 0.

    def inference(self, sequence):
        """Tags a sequence with part of speech tags.

        You should implement different kinds of inference (suggested as separate
        methods):

            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        # FOR Written-up
        # vocab = defaultdict(int)
        # for s in sequence:
        #     curword = s
        #     if self.rare:
        #         if curword not in self.vocab:
        #             if self.unk_suffix:
        #                 curword = self.find_unk_type(curword)
        #             else:
        #                 curword = 'rare'
        #     vocab[curword] += 1
        # print('inference',len(vocab))
        # exit()

        # convert unseen words
        if self.rare:
            for i in range(len(sequence)):
                curword = sequence[i]
                if curword not in self.vocab_set:
                    if self.unk_suffix:
                        curword = self.find_unk_type(curword)
                    else:
                        curword = 'rare'
                    sequence[i] = curword

        if self.inference_type == 'beam':
            decode = self.beam_search_decode(sequence)
        elif self.inference_type == 'greedy':
            # decode = self.greedy_decode(sequence)
            self.beam_k=1
            decode = self.beam_search_decode(sequence)
        elif self.inference_type == 'viterbi':
            decode = self.viterbi_decode(sequence)
        else: 
            print('Illigal decoding type')
            exit()
        
        return decode
    
    def beam_search_decode(self,seq): # with vectorization
        score_saver = np.full((len(seq)+1, self.tag_len, self.tag_len),float('-inf')) # \pi(i,yi)
        bp_saver = np.full((len(seq)+1, self.tag_len, self.tag_len),None) 
        score_saver[0,0,0] = math.log(self.transition_prob_mat[0,0,0]\
                                    * self.emission_prob_mat[0,0])
        stack = [(0,0)]
        start = time.time()

        for k in range(1, len(seq)+1):
            curword = seq[k-1]
            wordidx = self.reverse_vocab[curword]
            maxscore_dict = {}
            for v in range(self.tag_len):
                for u in range(self.tag_len):
                    max_score = float('-Inf')
                    max_tag = None
                    for w in range(self.tag_len):
                        if (w,u) not in stack:
                            continue
                        score = self.transition_prob_mat[w,u,v] \
                                + self.emission_prob_mat[v,wordidx] \
                                + score_saver[k-1,w,u]
                        if score > max_score:
                            max_score = score
                            max_tag = w
                        maxscore_dict[score] = [w,(u,v)]
                    score_saver[(k, u, v)] = max_score
                    bp_saver[(k, u, v)] = max_tag
            score_dict = sorted(maxscore_dict.items(),reverse=True)[:self.beam_k]
            stack = [score_dict[key][1][1] for key in range(len(score_dict))]

        res = []
        uidx, vidx, xidx = None, None, None
        maxscore = float('-inf')
        for u in range(self.tag_len):
            for v in range(self.tag_len):
                if score_saver[len(seq),u,v]>maxscore:
                    maxscore = score_saver[len(seq),u,v]
                    uidx = u
                    vidx = v
        res.append(self.tag_set[vidx])
        res.append(self.tag_set[uidx])

        for k in range(len(seq),2,-1):
            widx = bp_saver[k,uidx,vidx]
            res.append(self.tag_set[bp_saver[k,uidx,vidx]])
            uidx,vidx = widx, uidx
        # print(res)
        res=res[::-1]
        return res

    def viterbi_decode(self,seq):
        # if self.ngram =='4':
            # return self.viterbi_decode_4(seq)
        if self.ngram == '3':
            score_saver = np.full((len(seq)+1, self.tag_len, self.tag_len),float('-inf')) # \pi(i,yi)
            bp_saver = np.full((len(seq)+1, self.tag_len, self.tag_len),None) 
            score_saver[0,0,0] = math.log(self.transition_prob_mat[0,0,0]\
                                    * self.emission_prob_mat[0,0])
        elif self.ngram == '2':
            score_saver = np.full((len(seq)+1, self.tag_len),float('-inf')) # \pi(i,yi)
            bp_saver = np.full((len(seq)+1, self.tag_len),None) 
            score_saver[0,0] = math.log(self.transition_prob_mat[0,0]\
                                    * self.emission_prob_mat[0,0])
        elif self.ngram =='4':
            score_saver = np.full((len(seq)+1, self.tag_len,self.tag_len,self.tag_len),float('-inf')) # \pi(i,yi)
            bp_saver = np.full((len(seq)+1, self.tag_len,self.tag_len,self.tag_len),None) 
            score_saver[0,0,0,0] = math.log(self.transition_prob_mat[0,0,0,0]\
                                    * self.emission_prob_mat[0,0])
        start = time.time()
        
        for k in range(1, len(seq)+1):
            curword = seq[k-1]
            wordidx = self.reverse_vocab[curword]
            
            # print(self.transition_prob_mat.shape) (46, 46, 46)
            # print(self.emission_prob_mat[:,wordidx].shape) (46,)
            # trans(w,u,v) + emis(v,curword) + pi (k-1,w,u)
            scores = self.transition_prob_mat \
                    + self.emission_prob_mat[:,wordidx] # + np.array([score_saver[k-1,:,:]])
            if self.ngram == '3':
                for v in range(self.tag_len):
                    scores[:,:,v] += score_saver[k-1,:,:]
                score_saver[k,:,:]=scores.max(axis=0)
                bp_saver[k,:,:] = scores.argmax(axis=0)
            elif self.ngram == '2':
                for v in range(self.tag_len):
                    scores[:,v] += score_saver[k-1,:]
                score_saver[k,:]=scores.max(axis=0)
                bp_saver[k,:] = scores.argmax(axis=0)
            elif self.ngram == '4':
                for v in range(self.tag_len):
                    scores[:,:,:,v] += score_saver[k-1,:,:,:]
                score_saver[k,:,:,:]=scores.max(axis=0)
                bp_saver[k,:,:,:] = scores.argmax(axis=0)
        # print('unknown word cnt=%d'%unk_cnt)
                    
        res = []
        uidx, vidx, xidx = None, None, None
        maxscore = float('-inf')
        if self.ngram == '3':
            for u in range(self.tag_len):
                for v in range(self.tag_len):
                    if score_saver[len(seq),u,v]>maxscore:
                        maxscore = score_saver[len(seq),u,v]
                        uidx = u
                        vidx = v
            res.append(self.tag_set[vidx])
            res.append(self.tag_set[uidx])

            for k in range(len(seq),2,-1):
                widx = bp_saver[k,uidx,vidx]
                res.append(self.tag_set[bp_saver[k,uidx,vidx]])
                uidx,vidx = widx, uidx
        elif self.ngram == '2':
            for u in range(self.tag_len):
                if score_saver[len(seq),u]>maxscore:
                    maxscore = score_saver[len(seq),u]
                    uidx = u
            # res.append(self.tag_set[vidx])
            res.append(self.tag_set[uidx])

            for k in range(len(seq),1,-1):
                widx = bp_saver[k,uidx]
                res.append(self.tag_set[bp_saver[k,uidx]])
                uidx = widx
        elif self.ngram == '4':
            for u in range(self.tag_len):
                for v in range(self.tag_len):
                    for x in range(self.tag_len):
                        if score_saver[len(seq),u,v,x]>maxscore:
                            maxscore = score_saver[len(seq),u,v,x]
                            uidx = u
                            vidx = v
                            xidx = x
            res.append(self.tag_set[xidx])
            res.append(self.tag_set[vidx])
            res.append(self.tag_set[uidx])

            for k in range(len(seq),3,-1):
                widx = bp_saver[k,uidx, vidx, xidx]
                res.append(self.tag_set[bp_saver[k,uidx, vidx, xidx]])
                uidx,vidx,xidx = widx,uidx,vidx
        
        res = res[::-1]
        return res

    def find_unk_type(self,word):
        is_adj = [word.endswith(suf) for suf in self.suf_for_adj]
        # is_adj += [word.startswith(pre) for pre in self.pre_for_adj]
        flag = reduce(lambda i, j: int(i) | int(j), is_adj)
        if flag:
            return 'unk_jj'
        is_adv = [word.endswith(suf) for suf in self.suf_for_adv]
        if reduce(lambda i, j: int(i) | int(j), is_adv):
            return 'unk_adv'
        is_noun = [word.endswith(suf) for suf in self.suf_for_noun]
        if reduce(lambda i, j: int(i) | int(j), is_noun):
            return 'unk_nn'
        is_verb = [word.endswith(suf) for suf in self.suf_for_verb]
        if reduce(lambda i, j: int(i) | int(j), is_verb):
            return 'unk_vb'
        is_upper = [w.isupper() for w in word]
        if reduce(lambda i, j: int(i) | int(j),is_upper) or (word.capitalize()==word and len(word)>5):
            return 'unk_nnp'
        is_digit = [w.isdigit() for w in word]
        if reduce(lambda i, j: int(i) | int(j), is_digit):
            return 'unk_num'
        is_punc = [w in self.puncs for w in word]
        if reduce(lambda i, j: int(i) | int(j), is_punc):
            return 'unk_punc'
        return 'rare'


    def find_dicts(self,data): # "SS_word", "SS_Tag"
        if self.ngram == '2':
            interpolation_helper=None
        if self.ngram >= '3':
            ngram_uni_cnt = defaultdict(int)
        if self.ngram == '4':
            ngram_bi_cnt = defaultdict(int)

        words = list(data['word'])
        tags = list(data['tag'])
        vocab = defaultdict(int)
        n = int(self.ngram)
        tag_trans_cnt, emit_cnt, tag_num = defaultdict(lambda: defaultdict(int)),defaultdict(lambda: defaultdict(int)),defaultdict(int)
        # Number of transition from prev tag to current tag
        # Number of word with certain tag
        # number of current kind of tag

        ngram_cnt, ngram_prev_cnt = defaultdict(int), defaultdict(int)
         # if trigram, store count of three continous words and two continous words

        prevtag = tags[0]
        emit_cnt[tags[0]][words[0]] = 1
        tag_num[tags[0]] = 1
        vocab[words[0]] += 1

        ngram_cnt[tuple(['O' for i in range(n)])] = 1
        ngram_prev_cnt[tuple(['O' for i in range(n-1)])] = 1
        if self.ngram >= '3':
            ngram_uni_cnt[tuple(['O'])] = 1
        if self.ngram == '4':
            ngram_bi_cnt[tuple(['O','O'])] = 1
        
        for i in range(1, len(data)):
            vocab[words[i]]+=1

        # print('vocab size',len(vocab)) # 37505
        
        if self.rare:
            # vocab would include self.unk_list or 'rare'
            a = len(vocab)
            vocab = {a:b for a,b in vocab.items() if b>=self.rare_thr}
            b = len(vocab)
            print('%d words are marked rare'%(a-b))
            vocab = defaultdict(int,vocab)

        length = len(data)
        start = time.time()
        for i in range(1, len(data)):
            if i==1000000:
                break
            curword = words[i]
            if self.rare:
                if curword not in vocab:
                    if self.unk_suffix:
                        curword = self.find_unk_type(curword)
                    else:
                        curword = 'rare'
                    vocab[curword] += 1

            curtag = tags[i]
            # tag_trans_cnt[prevtag][curtag] += 1
            emit_cnt[curtag][curword] +=1
            tag_num[curtag] += 1
            # vocab[curword] += 1

            tmptag = find_conditional_window(n,i,tags,win_type='t')
            tag_trans_cnt[tuple(tmptag[:-1])][tmptag[-1]] += 1

            # tmp = find_conditional_window(n,i,words)
            tmp = tmptag
            if len(tmp)!=n: print("!!!len(tmp)!=n"); exit()
            ngram_cnt[tuple(tmp)] += 1
            ngram_prev_cnt[tuple(tmp[:n-1])] += 1
            if self.ngram >= '3':
                ngram_uni_cnt[tuple([tmp[-1]])] += 1
            if self.ngram == '4':
                ngram_bi_cnt[tuple(tmp[-2:])] += 1

        print('i:',i,'time used',time.time()-start)
        if self.ngram == '3':
            interpolation_helper = [ngram_uni_cnt]
        if self.ngram == '4':
            interpolation_helper = [ngram_uni_cnt, ngram_bi_cnt]    

        return vocab, tag_trans_cnt, emit_cnt, tag_num, ngram_cnt, ngram_prev_cnt, interpolation_helper
    

    def find_transition_prob_mat(self):
        if self.ngram == '2':
            transition_prob_mat = np.full((self.tag_len,self.tag_len),float('-inf')) # back pointer (i,yi)
        elif self.ngram == '3':
            transition_prob_mat = np.full((self.tag_len,self.tag_len, self.tag_len),float('-inf')) # back pointer (i,yi)
        elif self.ngram == '4':
            transition_prob_mat = np.full((self.tag_len,self.tag_len,self.tag_len, self.tag_len),float('-inf')) # back pointer (i,yi)
        reverse_tagset = {self.tag_set[k]:k for k in range(self.tag_len)}
        # print(reverse_tagset)

        # transition_prob_mat[curgram] = x
        trans_keys = self.ngram_cnt.keys()
        V = len(self.vocab.keys())
        for cur_ngram in trans_keys:
            if self.smooth_type=='addk':
                # transition_prob_mat[key[:-1]][key[-1]] = (self.ngram_cnt[key] + self.k) / (self.ngram_prev_cnt[key[:-1]] + self.k*V )
                val = (self.ngram_cnt[cur_ngram] + self.k) / (self.ngram_prev_cnt[cur_ngram[:-1]] + self.k*V )
                
            elif self.smooth_type=='interpolation':
                if self.ngram == '2': # curgram = w_{n-1},w{n}
                    features = np.array([self.ngram_cnt[cur_ngram]/self.ngram_prev_cnt[cur_ngram[:-1]], 
                                        self.ngram_prev_cnt[tuple([cur_ngram[-1]])]/V ])
                if self.ngram == '3':# curgram =w_{n-2}, w_{n-1},w{n}
                    ngram_uni_cnt = self.interpolation_helper[0]
                    features = np.array([self.ngram_cnt[cur_ngram]/self.ngram_prev_cnt[cur_ngram[:-1]],
                                        self.ngram_prev_cnt[cur_ngram[1:]]/ngram_uni_cnt[cur_ngram[1:-1]],
                                        ngram_uni_cnt[cur_ngram[-1]]/V ])
                if self.ngram == '4': # curgram = w_{n-3}, w_{n-2}, w_{n-1},w{n}
                    ngram_uni_cnt, ngram_bi_cnt = self.interpolation_helper
                    features = np.array([self.ngram_cnt[cur_ngram]/self.ngram_prev_cnt[cur_ngram[:-1]],
                                        self.ngram_prev_cnt[cur_ngram[1:]]/ngram_bi_cnt[cur_ngram[1:-1]],
                                        ngram_bi_cnt[cur_ngram[2:]]/ngram_uni_cnt[cur_ngram[2:-1]],
                                        ngram_uni_cnt[cur_ngram[-1]]/V ])
                val = sum(self.lambda_coef * features)
            elif self.smooth_type=='none':
                val = (self.ngram_cnt[cur_ngram]) / (self.ngram_prev_cnt[cur_ngram[:-1]])
            if self.ngram == '2':
                w,v = reverse_tagset[cur_ngram[0]],reverse_tagset[cur_ngram[1]]
                transition_prob_mat[(w,v)] = math.log(val)
            elif self.ngram == '3':
                w,v,u = reverse_tagset[cur_ngram[0]],reverse_tagset[cur_ngram[1]],reverse_tagset[cur_ngram[2]]
                transition_prob_mat[(w,v,u)] = math.log(val)
            elif self.ngram == '4':
                w,v,u,x = reverse_tagset[cur_ngram[0]],reverse_tagset[cur_ngram[1]],reverse_tagset[cur_ngram[2]],reverse_tagset[cur_ngram[3]]
                transition_prob_mat[(w,v,u,x)] = math.log(val)

        return transition_prob_mat


    def find_emission_prob_mat(self):
        # emission_prob_mat = defaultdict(lambda: defaultdict(int))
        emission_prob_mat = np.zeros((self.tag_len,self.V))

        V = len(self.vocab.keys())
        for t in range(self.tag_len):
            tag = self.tag_set[t]
            for w in range(self.V):
                word = self.vocab_set[w]
                val = (self.emit_cnt[tag][word] + self.k) / (self.tag_num[tag] + self.k*V)
                # emission_prob_mat[tag][word] = math.log(val)
                emission_prob_mat[t][w] = math.log(val)
        return emission_prob_mat

    
def find_conditional_window(n,i,words,win_type='t'):
    if win_type=='t':
        token = 'O'
    else: token = '-DOCSTART-'

    if i<(n-1):
        tmp = [token]*(n-i-1)+words[:i+1]
    elif words[i-2]==token and n>3:
        tmp = [token]*(n-3)+words[i-2:i+1]
    elif words[i-1]==token and n>2:
        tmp = [token]*(n-2)+words[i-1:i+1]
    elif words[i]==token and n>1:
        tmp = [token]*(n-1)+[words[i]]
    # elif words[i-2]=='.' and n>3:
    #     tmp = ['.']*(n-3)+words[i-2:i+1]
    # elif words[i-1]=='.' and n>2:
    #     tmp = ['.']*(n-2)+words[i-1:i+1]
    else:
        tmp = words[i-n+1:i+1]
    return tmp


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--Ngram', type=str, default='3', 
                            help='could be 2, 3, 4 for bi-, tri-, 4-gram')
    parser.add_argument('--smooth_type', type=str, default='addk', 
                            help='None, addk, or interpolation') # GoodTuring, KneserNey
    parser.add_argument('--addk_param',type=float, default=0.001, 
                            help='the value of k for add-k smoothing')
    parser.add_argument('--inference_type', type=str, default='viterbi', 
                            help='choose among [greedy, beam, viterbi]')
    parser.add_argument('--beam_k', type=int, default=2, 
                            help='value of k for Compute scores of k-best states, choose among [2, 3]')
    parser.add_argument('--save_dev', default=0, action='store_true',
                            help='whether to save the result on the dev dataset')
    parser.add_argument('--dev', default=1, action='store_true',
                            help='whether to evaluate tagger on the dev dataset')
    parser.add_argument('--test', default=0, action='store_true',
                            help = 'whether to test')
    parser.add_argument('--sep_sentence', default=0, action='store_true',
                            help='if True, split input dataset by sentence(.) and tag each sentence separately')
    parser.add_argument('--sep_doc', default=1, action='store_true',
                        help='if True, tag each document separately')
    parser.add_argument('--rare', default=1, action='store_true',
                        help='if True, replace infrequent words')
    parser.add_argument('--rare_thr', type=int, default=2,
                        help='mark word in the training set as rare if it appears less or equal to rare_thr')
    parser.add_argument('--unk_suffix', default=1, action='store_true',
                        help='deduct tag of rare words using prefix and suffix')
    
    
    args = parser.parse_args()
    if args.unk_suffix: 
        args.rare = 1
    print(args)

    pos_tagger = POSTagger(args)

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    # print(train_data.head())
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    words = list(train_data['word'])
    tags = list(train_data['tag'])
    for i in range(len(words)):
        if words[i] == '-DOCSTART-':
            if tags[i] != 'O':
                print(words[i],tags[i])
    
    pos_tagger.train(train_data)
    # print(pos_tagger.ngram_prev_cnt)
    
    # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc.
    if args.dev:
        result = evaluate(dev_data, pos_tagger)
        # if not args.save_dev:
            # args.save_dev = True
    folder = args.Ngram+'gram-'+args.smooth_type
    if args.smooth_type == 'addk':
        folder += str(args.addk_param)
    
    if args.sep_sentence:
        folder += '-sep_sentence'
    if args.sep_doc:
        folder += '-sep_doc'

    folder += '-'+'rare'+str(args.rare)+'-thr'+str(args.rare_thr)+'-' \
                +'unk_suffix'+str(args.unk_suffix)+'-' \
                + args.inference_type
    if args.inference_type=='beam':
        folder+=str(args.beam_k)
    # if args.test:
        # folder = folder
    if args.save_dev or args.test:
        if not os.path.exists(folder):
            os.mkdir(folder)
        f = open(os.path.join(folder,'args_log.txt'),'w')
        print(args,file = f)
        f.close()
            
    if args.save_dev:
        if not args.dev:
            result = evaluate(dev_data, pos_tagger)
        dev_path = os.path.join(folder,'pred_dev_y.csv')
        dev_id = list(dev_data['id'])
        dev_id = dev_id[:len(result)]

        pred_dev_df = pd.DataFrame({'id':dev_id,'tag':result})
        pred_dev_df.to_csv(dev_path, index = False)
        print('File ',dev_path,' created')


    print('ARGS: ',args)

    if args.test:
        print("############ test ####################")
    # Predict tags for the test set
        test_predictions = []
        # for sentence in test_data:
            # test_predictions.extend(pos_tagger.inference(sentence))
        test_seq = list(test_data['word'])
        test_id = list(test_data['id'])
        test_pred = pos_tagger.inference(test_seq)
        res_df = pd.DataFrame({'id':test_id,'tag':test_pred})
        
        test_path = os.path.join(folder,'test_y.csv')
        res_df.to_csv(test_path, index = False)
        print('File ',test_path,' created')
    
    # Write them to a file to update the leaderboard
    # to do

