from fastbpe.funcs import trainFastIds,tokenizeFastIds
import numpy as np
import regex as re




class Tokenizer:

    def __init__(self,special_tokens=None,split_pattern=None,final_vocab_size=258):

        self.base_vocab_size = None
        self.stoi = None
        self.itos = None
        self.final_vocab_size = final_vocab_size
        self.vocab = None
        self.merges = None
        self.special_tokens = special_tokens
        self.split_pattern = split_pattern
        self.special_pattern = '|'.join(map(re.escape, special_tokens))
        self._special_regex = re.compile(f'({self.special_pattern})')
        self._init_vocab()
        return
    
    def _init_vocab(self):
        self.stoi = {s:i+257 for i,s in enumerate(self.special_tokens)}
        self.itos = {i+257:s for i,s in enumerate(self.special_tokens)}
        self.base_vocab_size = 257+len(self.itos)

    def _base_decode(self,ids):
        parts = []
        for i in ids:
            if i ==0:
                parts.append('<|RESERVED|>'.encode('utf-8'))
            elif i < 257:
                parts.append(bytes([i-1]))
            else:
                parts.append(self.itos[i].encode('utf-8'))
        return b''.join(parts).decode('utf-8', errors='replace')
    
    def _base_encode(self,text):
        chunks = re.split(f'({self.special_pattern})', text)
        out = []
        for c in chunks:
            if c in self.special_tokens:
                out.append(self.stoi[c])
            elif len(c) > 0:
                out.extend(b + 1 for b in c.encode("utf-8"))
        return out
    
    def _base_encode_pad(self,text):
        # converts text to base token ids with 0 padding at split boundaries
        # chunks = re.split(f'({self.special_pattern})', text)
        chunks = self._special_regex.split(text)

        out = []
        for c in chunks:
            if c in self.special_tokens:
                out.append(self.stoi[c])
                out.append(0)
            elif len(c)>0:
                # words = re.findall(self.split_pattern,c)
                words = [w for w in c.split() if w]
                for w in words:
                    out.extend(b + 1 for b in w.encode("utf-8"))
                    out.append(0)
        return out
    
    def _base_encode_split(self,text):
        # converts text to base token ids with split locations
        # chunks = re.split(f'({self.special_pattern})', text)
        chunks = self._special_regex.split(text)

        out = []
        split_indices = [0]
        cur_idx = 0
        # split on special tokens first to prevent clash with regex
        for c in chunks:
            if c in self.special_tokens:
                out.append(self.stoi[c])
                cur_idx+=1
                split_indices.append(cur_idx)
            elif len(c)>0:
                # split with regex
                # words = re.findall(self.split_pattern,c)
                words = [w for w in c.split() if w]
                for w in words:
                    token_ids = [b + 1 for b in w.encode("utf-8")]
                    out.extend(token_ids)
                    cur_idx+=len(token_ids)
                    split_indices.append(cur_idx)
        return out,split_indices
    
    def train_bpe(self,text):
        # train a bpe vocabulary from text

        if self.final_vocab_size is None or self.final_vocab_size <= self.base_vocab_size or self.base_vocab_size is None:
            raise Exception("Check vocab sizes. should not be None and final should be > base.")

        # text to base ids with 0 inserted a chunk boundaries - no merges will occur across 0's
        ids = self._base_encode_pad(text)
        id_list = np.array(ids,dtype=np.int32)
        self.merges, self.vocab = trainFastIds(id_list,self.final_vocab_size,self.base_vocab_size)
        self.vocab_size = len(self.vocab)
        out = id_list[~np.isin(id_list, [-2, 0])]
        return out
    
    def encode(self,text):
        # takes text and performs merges to get bpe ids
        # note that unlike _train_bpe, there are no zeros as boundaries for the c++ function. these are supplied as split idxs
        base_ids,splits = self._base_encode_split(text)
        base_ids = np.array(base_ids,dtype=np.int32)
        splits = np.array(splits,dtype=np.int32)
        result = tokenizeFastIds(base_ids, splits, self.merges, self.base_vocab_size)
        return result

    def _bpe_decode(self,ids):
        out = []
        for idx in ids:
            out.extend(self.vocab[idx])
        return out
    
    def decode(self,ids):
        base_ids = self._bpe_decode(ids)
        text = self._base_decode(base_ids)
        return text
    
    