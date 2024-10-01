import re
import torch
from torch import Tensor
import torch.nn as nn
from typing import Sequence
class CharsetAdapter:

    def __init__(self):
        super().__init__()
        self.charset = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.unsupported = re.compile(f'[^{re.escape(self.charset)}]')

    def __call__(self, label):
        label = label.upper()
        label = self.unsupported.sub('', label)
        return label


class Vocab:
    def __init__(self, charset):
        self.c2i = dict()
        self.c2i['<EOS>'] = 0
        count = 1
        for c in charset:
            if c not in self.c2i.keys():
                self.c2i[c] = count
                count += 1
        self.c2i['<SOS>'] = len(self.c2i)
        self.c2i['<PAD>'] = len(self.c2i)
        
        self.i2c = {v : k for k, v in self.c2i.items()}
    
    def __len__(self):
        return len(self.c2i)
    

class Tokenizer:
    def __init__(self, charset, max_len):
        
        self.max_len = max_len
        self.vocab = Vocab(charset)
        self.sos_id = self.vocab.c2i['<SOS>']
        self.eos_id = self.vocab.c2i['<EOS>']
        self.pad_id = self.vocab.c2i['<PAD>']
        self.special = [self.sos_id, self.eos_id, self.pad_id]
    
    def __len__(self):
        return len(self.vocab)
    
    def _tok2ids(self, token : str):
        return [self.vocab.c2i[c] for c in token]
    
    def _ids2tok(self, token_ids, join = True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        token = [self.vocab.i2c[i] for i in token_ids if i not in self.special]
        return ''.join(token) if join else token
    
    
    
    def encode_batch(self, labels : list[str], device):
        encoded_labels = []
        for label in labels:
            encoded_label = [self.sos_id] + self._tok2ids(label) + [self.eos_id]
            if len(encoded_label) > self.max_len:
                encoded_label = encoded_label[ : self.max_len]
            else:
                encoded_label = encoded_label + [self.pad_id] * (self.max_len - len(encoded_label))
            encoded_labels.append(torch.tensor(encoded_label, dtype = torch.long, device = device))
        return torch.stack(encoded_labels, dim = 0)
    
    def _filter(self, probs : Tensor, ids : Tensor):
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)
        ids = ids[ : eos_idx]
        probs = probs[: eos_idx + 1]
        return probs, ids
    
    def decode(self, token_dists : Tensor, raw : bool = False):

        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)
            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs


def init_weights(module: nn.Module, name: str = '', exclude: Sequence[str] = ()):
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)