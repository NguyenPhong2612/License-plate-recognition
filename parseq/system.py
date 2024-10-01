import torch
import torch.nn as nn
from timm.models.helpers import named_apply
from functools import partial
from .module import Encoder, Decoder, TokenEmbedding
from .utils import init_weights
import pytorch_lightning as pl
from .utils import Tokenizer, CharsetAdapter
import numpy as np
import math
from torch.optim import Optimizer
from timm.optim import create_optimizer_v2
from torch.optim.lr_scheduler import OneCycleLR
from itertools import permutations
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PARSeq(nn.Module):

    def __init__(self, config, device = device):
        super().__init__()

        self.max_len = config['model']['max_len']
        self.decode_ar = config['model']['decode_ar']
        self.refine_iters = config['model']['refine_iter']
        self.embed_dim = config['model']['d_model']
        self.num_tokens = config['model']['num_tokens']
        self.dropout = 0.1
        self.encoder = Encoder(config['model']['image_size'], config['model']['patch_size'], embed_dim = config['model']['d_model'], depth = config['model']['enc_depth'], num_heads = config['model']['enc_num_heads'], mlp_ratio = config['model']['enc_mlp_ratio'])
        self.decoder = Decoder(config)
        self.text_embed = TokenEmbedding(config)
        self.head = nn.Linear(self.embed_dim, self.num_tokens - 2)

        self.pos_queries = nn.Parameter(torch.Tensor(1, self.max_len + 1, self.embed_dim))
        self.dropout = nn.Dropout(self.dropout)
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_queries, std = 0.02)
        self._device = device
    
    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {'text_embed.embedding.weight', 'pos_queries'}
        enc_param_names = {'encoder.' + n for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)

    def encode(self, img: torch.Tensor):
        return self.encoder(img.to(self._device))

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask = None,
        tgt_padding_mask = None,
        tgt_query = None,
        tgt_query_mask = None):
        N, L = tgt.shape
        null_ctx = self.text_embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, : L - 1] + self.text_embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)

    def forward(self, tokenizer: Tokenizer, images, max_length):
        testing = max_length is None
        max_length = self.max_len if max_length is None else min(max_length, self.max_len)
        bs = images.shape[0]
        num_steps = max_length + 1
        memory = self.encode(images).to(device)
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        tgt_mask = query_mask = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=self._device), 1)

        if self.decode_ar:
            tgt_in = torch.full((bs, num_steps), tokenizer.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = tokenizer.sos_id

            logits = []
            for i in range(num_steps):
                j = i + 1  
                tgt_out = self.decode(
                    tgt_in[:, :j],
                    memory,
                    tgt_mask[:j, :j],
                    tgt_query = pos_queries[:, i:j],
                    tgt_query_mask = query_mask[i:j, :j],)

                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < num_steps:
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    if testing and (tgt_in == tokenizer.eos_id).any(dim=-1).all():
                        break

            logits = torch.cat(logits, dim=1)
        else:
            tgt_in = torch.full((bs, 1), tokenizer.sos_id, dtype=torch.long, device=self._device)
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)

        if self.refine_iters:
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), tokenizer.sos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = (tgt_in == tokenizer.eos_id).int().cumsum(-1) > 0
                tgt_out = self.decode(
                    tgt_in, memory, tgt_mask, tgt_padding_mask, pos_queries, query_mask[:, : tgt_in.shape[1]])
                logits = self.head(tgt_out)

        return logits
    


class System(pl.LightningModule):

    def __init__( self, config):
        
        super().__init__()
        self.save_hyperparameters()
        self.max_len = int(config['model']['max_len'])
        self.charset_adapter = CharsetAdapter()
        self.charset = config['model']['train_charset']
        self.lr = float(config['trainer']['lr'])
        self.batch_size = config['trainer']['batch_size']
        self.warm_pct = float(config['trainer']['warm_pct'])
        self.weight_decay = float(config['trainer']['weight_decay'])
        self.tokenizer = Tokenizer(self.charset, self.max_len)
        self.sos_id = self.tokenizer.sos_id
        self.eos_id = self.tokenizer.eos_id
        self.pad_id = self.tokenizer.pad_id
        
        self.model = PARSeq(config)
        self.rng = np.random.default_rng()
        self.max_gen_perms = config['model']['perm_num'] // 2 if config['model']['perm_mirrored'] else config['model']['perm_num']
        self.perm_forward = config['model']['perm_forward']
        self.perm_mirrored = config['model']['perm_mirrored']
        if config['model']['pretrained']:
            self.weight_ulr = config['model']['weight_url']
            self.load_weight(self.weight_ulr)
        self.set_seed()
        
    def set_seed(self, seed = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        
    def configure_optimizers(self):
        agb = self.trainer.accumulate_grad_batches
        lr_scale = agb * math.sqrt(self.trainer.num_devices) * self.batch_size / 256.0
        lr = float(lr_scale) * float(self.lr)
        optim = create_optimizer_v2(self, 'adamw', lr, self.weight_decay)
        sched = OneCycleLR(
            optim, lr, self.trainer.estimated_stepping_batches, pct_start=self.warm_pct, cycle_momentum=False
        )
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}
    
    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)
    
    def forward(self, images, max_length = None):
        return self.model.forward(self.tokenizer, images, max_length)

    def gen_tgt_perms(self, tgt):
        max_num_chars = tgt.shape[1] - 2
        if max_num_chars == 1:
            return torch.arange(3, device=self._device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=self._device)] if self.perm_forward else []
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
       
        if max_num_chars < 5:
            
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(
                list(permutations(range(max_num_chars), max_num_chars)),
                device=self._device,
            )[selector]
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend(
                [torch.randperm(max_num_chars, device = self._device) for _ in range(num_gen_perms - len(perms))]
            )
            perms = torch.stack(perms)
        if self.perm_mirrored:
            comp = perms.flip(-1)
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        sos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([sos_idx, perms + 1, eos_idx], dim=1)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)
        return perms

    def generate_attn_masks(self, perm):
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), dtype=torch.bool, device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1 :]
            mask[query_idx, masked_keys] = True
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = True  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.to(device)
        tgt = labels.to(device)

        memory = self.model.encode(images.to(device))

        tgt_perms = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)

        loss = 0
        loss_numel = 0
        n = (tgt_out != self.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm)
            out = self.model.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
            logits = self.model.head(out).flatten(end_dim=1)
            loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
            loss_numel += n
            if i == 1:
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                n = (tgt_out != self.pad_id).sum().item()
                
        loss /= loss_numel
        with torch.no_grad():
            self.eval()
            logits, _, _ = self.forward_logits_loss(images, labels)
        predicted_labels, _ = self.tokenizer.decode(logits.softmax(-1))
        predicted_labels = [self.charset_adapter(label) for label in predicted_labels]
        true_labels = self.decode(labels)
        count = 0
        for i in range(len(true_labels)):
            if true_labels[i] == predicted_labels[i]:
                count += 1
        train_acc = float(count / len(true_labels))
        self.log("train_loss", loss, on_epoch = True, prog_bar = True, logger = True)
        self.log("train_acc", train_acc, on_epoch = True, prog_bar = True, logger = True)
        return loss
    
    def forward_logits_loss(self, images, targets: list[str]):
        targets = targets[:, 1:] 
        max_len = targets.shape[1] - 1
        logits = self.forward(images, max_len)
        loss = F.cross_entropy(logits.flatten(end_dim = 1), targets.flatten(), ignore_index = self.pad_id)
        loss_numel = (targets != self.pad_id).sum()
        return logits, loss, loss_numel
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        images, labels = batch
        with torch.no_grad():
            logits, loss, loss_numel = self.forward_logits_loss(images, labels)
            predicted_labels, _ = self.tokenizer.decode(logits.softmax(-1))
            predicted_labels = [self.charset_adapter(label) for label in predicted_labels]
            true_labels = self.decode(labels)
            count = 0
            for i in range(len(true_labels)):
                if true_labels[i] == predicted_labels[i]:
                    count += 1
            val_acc = float(count / len(true_labels))
            self.log("val_loss", loss / loss_numel, on_epoch = True, prog_bar = True, logger = True)
            self.log("val_acc", val_acc, on_epoch = True, prog_bar = True, logger = True)

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics["train_loss"].item()
        train_acc = self.trainer.callback_metrics["train_acc"].item()
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        val_acc = self.trainer.callback_metrics["val_acc"].item()
        combined_acc = val_acc + 1e-1 * train_acc
        self.log("combined_acc", combined_acc, prog_bar = False, logger = True)
        print(f"Epoch {self.current_epoch}: train_loss = {train_loss:.3f}, train_acc = {train_acc:.3f}, val_loss = {val_loss:.3f}, val_acc = {val_acc:.3f}")
    
    def load_weight(self, url):
        state_dict = torch.hub.load_state_dict_from_url(url = url, map_location = 'cuda', check_hash = True)
        self.model.load_state_dict(state_dict)
        print("Load weights sucessfully !!!")
    
    def decode(self, ids):
        true_labels = []
        if isinstance(ids):
            ids = ids.tolist()
        for label in ids:
            true_label = self.tokenizer._ids2tok(label)
            true_labels.append(self.charset_adapter(true_label))
        return true_labels