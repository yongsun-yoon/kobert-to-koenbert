import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def minilm_loss_fn(s, t, num_heads, attention_mask=None, temperature=1.0):
    attn_t = attention(t, t, num_heads, attention_mask)
    attn_s = attention(s, s, num_heads, attention_mask)
    loss = kl_div_loss(attn_s, attn_t, temperature=temperature)
    return loss



def kl_div_loss(s, t, temperature):
    if len(s.size()) != 2:
        s = s.view(-1, s.size(-1))
        t = t.view(-1, t.size(-1))

    s = F.log_softmax(s / temperature, dim=-1)
    t = F.softmax(t / temperature, dim=-1)
    return F.kl_div(s, t, reduction='batchmean') * (temperature ** 2)


def transpose_for_scores(h, num_heads):
    batch_size, seq_length, dim = h.size()
    head_size = dim // num_heads
    h = h.view(batch_size, seq_length, num_heads, head_size)
    return h.permute(0, 2, 1, 3) # (batch, num_heads, seq_length, head_size)


def attention(h1, h2, num_heads, attention_mask=None):
    # assert h1.size() == h2.size()
    head_size = h1.size(-1) // num_heads
    h1 = transpose_for_scores(h1, num_heads) # (batch, num_heads, seq_length, head_size)
    h2 = transpose_for_scores(h2, num_heads) # (batch, num_heads, seq_length, head_size)

    attn = torch.matmul(h1, h2.transpose(-1, -2)) # (batch_size, num_heads, seq_length, seq_length)
    attn = attn / math.sqrt(head_size)
    if attention_mask is not None:
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1 - attention_mask) * -10000.0
        attn = attn + attention_mask
    return attn


def to_distill(model):
    model.base_model.encoder.layer[0].attention.self.__class__._forward = bert_self_attention_forward
    for layer in model.base_model.encoder.layer:
        layer.attention.self.forward = layer.attention.self._forward
    return model


def bert_self_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_value=None,
    output_attentions=False,
):
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)

    self.q = mixed_query_layer # (bs, sq, dim)
    self.k = mixed_key_layer # (bs, sq, dim)
    self.v = mixed_value_layer # (bs, sq, dim)
    
    query_layer = self.transpose_for_scores(mixed_query_layer) # (bs, nh, sq, dim)
    key_layer = self.transpose_for_scores(mixed_key_layer) # (bs, nh, sq, dim)
    value_layer = self.transpose_for_scores(mixed_value_layer) # (bs, nh, sq, dim)
    
    if self.is_decoder:
        past_key_value = (key_layer, value_layer)

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # (bs, nh, sq, sq)
    self.attn = attention_scores

    if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        seq_length = hidden_states.size()[1]
        position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
        distance = position_ids_l - position_ids_r
        positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
        positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

        if self.position_embedding_type == "relative_key":
            relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores
        elif self.position_embedding_type == "relative_key_query":
            relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = nn.Softmax(dim=-1)(attention_scores)
    attention_probs = self.dropout(attention_probs)

    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    self.o = context_layer

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    if self.is_decoder:
        outputs = outputs + (past_key_value,)
    return outputs
