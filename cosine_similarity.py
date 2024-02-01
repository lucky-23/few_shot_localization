#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March  1 13:16:02 2023

@author: kanchan
"""

import torch
import torch.nn.functional as F
from torch import nn

# Define a transformer encoder layer with cross-attention
class CrossAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=4, dim_feedforward=512, dropout=0.1):
        super(CrossAttentionEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, key=None, value=None, query=None):
        key = key if key is not None else src
        value =value if value is not None else src
        query = query if query is not None else query

        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
       
        cross_attn_output = self.cross_attn(query, key, value)[0]
     
        src = src + self.dropout(cross_attn_output)
        src = self.norm2(src)

        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        return src

# Define the transformer model with cross-attention for each video
class CrossAttentionTwoEncoderModel(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1):
        super(CrossAttentionTwoEncoderModel, self).__init__()
        self.encoder1_layers = nn.ModuleList([
            CrossAttentionEncoderLayer(d_model, nhead, dim_feedforward, dropout) if i > 0 else nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for i in range(num_layers)
        ])

        self.encoder2_layers = nn.ModuleList([
            CrossAttentionEncoderLayer(d_model, nhead, dim_feedforward, dropout) if i > 0 else nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for i in range(num_layers)
        ])

    def forward(self, x1, x2):
        encoder1_outputs = []
        encoder2_outputs = []

        for i, (encoder1_layer, encoder2_layer) in enumerate(zip(self.encoder1_layers, self.encoder2_layers)):
            if i > 0:
               
                query1 = F.cosine_similarity(x1, encoder2_outputs[-1], dim=-3).unsqueeze(1)
                x1 = encoder1_layer(x1, key=encoder2_outputs[-1], value=encoder2_outputs[-1], query=query1)
               
                query2 = F.cosine_similarity(x2, encoder1_outputs[-1], dim=-3).unsqueeze(1)
                x2 = encoder2_layer(x2, key=encoder1_outputs[-1], value=encoder1_outputs[-1], query=query2)
               
            else:
                x1 = encoder1_layer(x1)
                x2 = encoder2_layer(x2)

            encoder1_outputs.append(x1)
            encoder2_outputs.append(x2)
          
        # Calculate final cosine similarity between encoder1_outputs and encoder2_outputs
        final_cosine_similarity = F.cosine_similarity(encoder1_outputs[-1], encoder2_outputs[-1], dim=-3)
           
        return final_cosine_similarity

# Create an instance of the transformer model with cross-attention for each video
cross_attention_two_encoder_model = CrossAttentionTwoEncoderModel()
#print("111111",cross_attention_two_encoder_model.shape)
# Assuming you have video features x1 and x2 for each video of shape bxtxd
x1 = torch.rand(1, 128, 256)  # Example shape, adjust accordingly
x2 = torch.rand(1, 128, 256)  # Example shape, adjust accordingly

# Pass x1 and x2 through the transformer model with cross-attention
final_cosine_similarity = cross_attention_two_encoder_model(x1, x2)
#print("final_similairty",final_cosine_similarity.shape)

#print("Final Cosine Similarity:", final_cosine_similarity.item())