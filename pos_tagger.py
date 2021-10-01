""" Contains the part of speech tagger class. """
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score
import time
import math

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
    seq = seq[100:200]
    true_tag = true_tag[100:200]
    # seq = ['-DOCSTART-', 'I', 'am', 'Sam','.']
    # ['O','PRP','VBP','NNP','.']
    # seq = np.array(seq)
    # seq_idx = np.where(seq=='.')
    # seq_idx = np.concatenate(([0],seq_idx[0]))
    # res = []
    # for i in range(1,len(seq_idx)):
    #     part_seq = seq[seq_idx[i-1]:seq_idx[i]+1]
    #     print(part_seq)
    #     # res += model.inference(part_seq)
    # exit()

    model.inference(seq)
    res = model.res
    # res = model.inference(seq)
    print('res',res)
    print('true_tag',true_tag)
    acc = accuracy_score(res,true_tag)
    print('acc',acc)
    pass


class POSTagger():
    def __init__(self,args):
        """Initializes the tagger model parameters and anything else necessary. """
        self.suf_for_adj = ['sy', 'dy', 'ic', 'ese', 'esque', 'al', 'able', 'ive', 'ish', 'ous', 'zy', 'less', 'ical', 'ly', 'ian', 'ible', 'ful', 'lly', 'i']
        self.suf_for_adv = ['ally', 'wards', 'ily', 'wise', 'ward', 'ly']
        self.suf_for_noun = ['ity', 'hood', 'ment', 'al', 'ness', 'acy', 'dom', 'ling', 'ty', 'or', 'ation', 'ship', 'ry', 'ery', 'cy', 'ee', 'age', 'ist', 'ism', 'er', 'action', 'scape', 'ure', 'ion', 'ance', 'ence']
        self.suf_for_verb  = ['ify', 'fy', 'ate', 'en', 'ize', 'ise']

        self.ngram = args.Ngram
        self.smooth_type = args.smooth_type
        if self.smooth_type=='addk':
            self.k = 0.001 # .5? .05? .01? # bigger, then the trans/emit prob is more far from true val
        if self.smooth_type=='interpolation':
            if self.ngram == '2':
                self.lambda_coef = [0.8,0.2]
            elif self.ngram == '3':
                self.lambda_coef = [0.8,0.1,0.1]
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
                    = self.find_dicts(data) #, self.ngram_cnt, self.ngram_prev_cnt\
                    

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

        decode = self.viterbi_decode(sequence)
        return []

    def viterbi_decode(self,seq):
        # seq = ['-DOCSTART-', 'I', 'am', 'Sam','.']
    # ['O','PRP','VBP','NNP','.']
        n = self.ngram
        tag_set = list(self.tag_num.keys())
        tag_len = len(tag_set)
        state_set = list(self.ngram_prev_cnt.keys())
        state_len = len(state_set)
        score_saver = np.zeros((state_len, len(seq))) # \pi(i,yi)
        bp_saver = np.full((state_len, len(seq)),None) # back pointer (i,yi)
        
        reverse_stateset = {state_set[k]:k for k in range(state_len)}
        # print(reverse_stateset) # -> {('VBZ', 'JJR'): 575}
        # print(state_set[575]) # -> ('VBZ', 'JJR')

        # when time step = 0 
        for m in range(state_len):
            curtag = state_set[m] + (START_TAG,)
            # if state_set[m] == (START_TAG, START_TAG):
            #     score_saver[m][0] = 1
            # else: 
            #     score_saver[m][0] = 0
            emitp = self.find_emission_prob(START_TAG,START_SYM)
            transp = self.find_transition_prob(curtag)
            score_saver[m][0] = math.log(emitp*transp)
            bp_saver[m][0] = 0

        # when time step = 1 to n
        for i in range(1,len(seq)): # i
            curword = seq[i]
            print(curword)
            for j in range(state_len):  # yi = j
                curtag = state_set[j]
                maxscore = float("-inf")
                bestprev = None
                for k in range(tag_len):
                    prevtag = tag_set[k]
                    curgram = (prevtag,)+curtag
                    if curgram[:-1] not in state_set:
                        continue
                    prev_idx = reverse_stateset[curgram[:-1]] # y_{i-1}
                    emitp = self.find_emission_prob(curgram[-1],curword)
                    transp = self.find_transition_prob(curgram)
                    # cur_score = math.log(emitp) * math.log(transp) * score_saver[prev_idx][i-1]
                    cur_score = math.log(emitp*transp) + score_saver[prev_idx][i-1]
                    # print(curgram)
                    # print('emitp',emitp,'curgram[-1]',curgram[-1],'curword',curword)
                    # print('transp',transp,'curgram',curgram)
                    # print('cur_score',cur_score)
                    if cur_score > maxscore:
                        maxscore = cur_score
                        bestprev = prev_idx # tag state can be found by state_set[prev_idx]
                score_saver[j][i] = maxscore
                bp_saver[j][i] = bestprev
                # print('curtag',state_set[j],'best score with prev tag',state_set[bestprev], 'score',maxscore)
            # maxval = max(score_saver[:,i])
            # row_idx = np.where(score_saver[:,i]==maxval)[0]
            # lastptr = bp_saver[row_idx,i][0]
            # print(state_set[lastptr])
            # break

        res = []
        maxval = max(score_saver[:,-1])
        row_idx = np.where(score_saver[:,-1]==maxval)[0][0]
        cnt = len(seq)-1
        while cnt>=0:
            # print(cnt, state_set[row_idx])
            t = state_set[row_idx][-1]
            res.append(t)
            if t == 'O':
                break
            lastptr = bp_saver[row_idx,cnt]
            row_idx = lastptr
            cnt-=1
        res = res[::-1]
        # print(res)
        self.res = res
        # return res
    
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

        # ngram_cnt[tuple(['-DOCSTART-' for i in range(n)])] = 1
        # ngram_prev_cnt[tuple(['-DOCSTART-' for i in range(n-1)])] = 1
        ngram_cnt[tuple(['O' for i in range(n)])] = 1
        ngram_prev_cnt[tuple(['O' for i in range(n-1)])] = 1
        if self.ngram >= '3':
            ngram_uni_cnt[tuple(['O'])] = 1
        if self.ngram == '4':
            ngram_bi_cnt[tuple(['O','O'])] = 1

        length = len(data)
        start = time.time()
        for i in range(1, len(data)):
            if i==1000000:
                break
            curword = words[i]
            curtag = tags[i]
            # tag_trans_cnt[prevtag][curtag] += 1
            emit_cnt[curtag][curword] +=1
            tag_num[curtag] += 1
            vocab[curword] += 1

            tmptag = find_conditional_window(n,i,tags,win_type='t')
            tag_trans_cnt[tuple(tmptag[:-1])][tmptag[-1]] += 1

            # tmp = find_conditional_window(n,i,words)
            tmp = tmptag
            if len(tmp)!=n: print("!!!len(tmp)!=n"); exit()
            ngram_cnt[tuple(tmp)] += 1
            ngram_prev_cnt[tuple(tmp[:n-1])] += 1
            if self.ngram >= '3':
                ngram_uni_cnt[tuple(tmp[-1])] += 1
            if self.ngram == '4':
                ngram_bi_cnt[tuple(tmp[-2:])] += 1

        print('i:',i,'time used',time.time()-start)
        if self.ngram == '3':
            interpolation_helper = [ngram_uni_cnt]
        if self.ngram == '4':
            interpolation_helper = [ngram_uni_cnt, ngram_bi_cnt]
        return vocab, tag_trans_cnt, emit_cnt, tag_num, ngram_cnt, ngram_prev_cnt, interpolation_helper

    def find_transition_prob(self,cur_ngram): # e.g. cur_ngram = ('NN','NP','.') for tri-gram
        V = len(self.vocab.keys())
        if self.smooth_type=='addk':
            res = (self.ngram_cnt[cur_ngram] + self.k) / (self.ngram_prev_cnt[cur_ngram[:-1]] + self.k*V )
        if self.smooth_type=='interpolation':
            if self.ngram == '2': # curgram = w_{n-1},w{n}
                features = np.array([self.ngram_cnt[cur_ngram]/self.ngram_prev_cnt[cur_ngram[:-1]], 
                                    self.ngram_prev_cnt[cur_ngram[-1]]/V ])
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
            res = sum(self.lambda_coef * features)
        return res
        
    def find_emission_prob(self,tag,word):
        V = len(self.vocab.keys())
        # if self.smooth_type=='addk':
        res = (self.emit_cnt[tag][word] + self.k) / (self.tag_num[tag] + self.k*V)
        if word not in self.vocab:
            print('self.emit_cnt[tag][word]',self.emit_cnt[tag][word])
            print()
            print("Unknown Word!")
            exit()
            
        # if self.smooth_type=='interpolation': # ???
            # res = (self.emit_cnt[tag][word]) / (self.tag_num[tag])
        return res

    
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
    # parser.add_argument('--Ngram', type=str, default='',
    #                         help='')
    parser.add_argument('--Ngram', type=str, default='3', 
                            help='could be 2, 3, 4 for bi-, tri-, 4-gram')
    parser.add_argument('--smooth_type', type=str, default='addk', 
                            help='addk, interpolation, GoodTuring, KneserNey')
    args = parser.parse_args()

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
    
    # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc.
    evaluate(dev_data, pos_tagger)

    exit()

    # Predict tags for the test set
    test_predictions = []
    for sentence in test_data:
        test_predictions.extend(pos_tagger.inference(sentence))
    
    # Write them to a file to update the leaderboard
    # to do

