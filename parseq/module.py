from timm.models.vision_transformer import PatchEmbed, VisionTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Encoder(VisionTransformer):

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed):
        super().__init__(
            image_size,
            patch_size,
            in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            embed_layer=embed_layer,
            num_classes=0,  
            global_pool='',  
            class_token=False)

    def forward(self, x):
        return self.forward_features(x.to(device))
    
class DecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.d_model = config['model']['d_model']
        self.dec_num_heads = config['model']['dec_num_heads']
        self.d_ff = config['model']['dec_mlp_ratio'] * self.d_model
        self.eps = 1e-5
        self.self_attn = nn.MultiheadAttention(self.d_model, self.dec_num_heads, dropout = 0.1, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(self.d_model, self.dec_num_heads, dropout = 0.1, batch_first=True)

        self.linear1 = nn.Linear(self.d_model, self.d_ff)
        self.dropout = nn.Dropout(p = 0.1)
        self.linear2 = nn.Linear(self.d_ff, self.d_model)

        self.norm1 = nn.LayerNorm(self.d_model, eps=self.eps)
        self.norm2 = nn.LayerNorm(self.d_model, eps=self.eps)
        self.norm_q = nn.LayerNorm(self.d_model, eps=self.eps)
        self.norm_c = nn.LayerNorm(self.d_model, eps=self.eps)
        self.dropout1 = nn.Dropout(p = 0.1)
        self.dropout2 = nn.Dropout(p = 0.1)
        self.dropout3 = nn.Dropout(p = 0.1)


    def forward_stream(
        self,
        tgt,
        tgt_norm,
        tgt_kv,
        memory,
        tgt_mask,
        tgt_key_padding_mask):

        tgt2, sa_weights = self.self_attn(
            tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)

        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights
    
    def forward(
        self,
        query,
        content,
        memory,
        query_mask = None,
        content_mask = None,
        content_key_padding_mask = None,
        update_content: bool = True):
        
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(query, query_norm, content_norm, memory, query_mask, content_key_padding_mask)[0]
        if update_content:
            content = self.forward_stream(
                content, content_norm, content_norm, memory, content_mask, content_key_padding_mask
            )[0]
        return query, content
    


class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, config):
        super().__init__()
        self.d_model = config['model']['d_model']
        self.num_layers = config['model']['dec_depth']
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(self.num_layers)])
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, query, content, memory, query_mask = None, content_mask = None, content_key_padding_mask = None):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(
                query, content, memory, query_mask, content_mask, content_key_padding_mask, update_content = not last)
        query = self.norm(query)
        return query


class TokenEmbedding(nn.Module):
    def __init__(self,  config):
        super().__init__()
        self.num_tokens = config['model']['num_tokens']
        self.d_model = config['model']['d_model']
        self.embedding = nn.Embedding(self.num_tokens, self.d_model)
    
    def forward(self, tokens):
        return math.sqrt(self.d_model) * self.embedding(tokens)
