"""
models.py - Seq2Seq Models with Attention (STARTER VERSION)

This module contains:
- Encoder (from Lesson 4 - PROVIDED)
- Decoder (from Lesson 4 - PROVIDED)
- AttentionLayer (NEW - TODO: You implement this!)
- DecoderWithAttention (NEW - TODO: You implement this!)
- Seq2SeqWithAttention (NEW - TODO: You implement this!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# ============================================================================
# FROM LESSON 4 - These are provided complete
# ============================================================================

class Encoder(nn.Module):
    """
    Bidirectional LSTM Encoder (From Lesson 4).
    This is provided complete - no changes needed!
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, padding_idx=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor (batch_size, seq_len)
            
        Returns:
            hidden: Final hidden state (1, batch_size, hidden_dim)
            cell: Final cell state (1, batch_size, hidden_dim)
        """
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # Combine bidirectional states
        hidden = torch.tanh(
            self.fc_hidden(torch.cat([hidden[-2], hidden[-1]], dim=1))
        ).unsqueeze(0)
        
        cell = torch.tanh(
            self.fc_cell(torch.cat([cell[-2], cell[-1]], dim=1))
        ).unsqueeze(0)
        
        return hidden, cell


# ============================================================================
# NEW FOR LESSON 5 - You need to implement these!
# ============================================================================

class AttentionLayer(nn.Module):
    """
    Bahdanau Attention Mechanism.
    
    TODO: Implement the forward method following the steps below.
    
    The attention mechanism computes:
    1. Alignment scores between decoder hidden state and encoder outputs
    2. Attention weights (softmax of scores)
    3. Context vector (weighted sum of encoder outputs)
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        # TODO: Define the attention network layers
        # Hint: You need two linear layers:
        #   1. self.attn: Maps concatenated [hidden*2] to hidden_dim
        #   2. self.v: Projects hidden_dim to scalar (1)
        
        # TODO: Uncomment and fill in the dimensions
        # self.attn = nn.Linear(???, ???)
        # self.v = nn.Linear(???, ???, bias=False)
        
        raise NotImplementedError("TODO: Initialize attention layers")
    
    def forward(self, hidden, encoder_outputs):
        """
        Compute attention weights and context vector.
        
        Args:
            hidden: Decoder hidden state (batch_size, hidden_dim)
            encoder_outputs: All encoder hidden states (batch_size, src_len, hidden_dim*2)
            
        Returns:
            context: Attention context vector (batch_size, hidden_dim*2)
            attention_weights: Attention distribution (batch_size, src_len)
        
        TODO: Implement the attention mechanism following these steps:
        
        Step 1: Repeat decoder hidden state for all source positions
                Use unsqueeze(1) and repeat(1, src_len, 1)
        
        Step 2: Concatenate with encoder outputs
                Use torch.cat(..., dim=2)
        
        Step 3: Compute energy (attention scores)
                Apply: energy = tanh(self.attn(combined))
        
        Step 4: Project to scalar scores
                Apply: scores = self.v(energy).squeeze(2)
        
        Step 5: Apply softmax to get attention weights
                Use F.softmax(..., dim=1)
        
        Step 6: Compute context as weighted sum
                Use torch.bmm with unsqueeze
        """
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # TODO: Step 1 - Repeat decoder hidden state
        # hidden should become (batch_size, src_len, hidden_dim)
        # hidden_repeated = ???
        
        # TODO: Step 2 - Concatenate with encoder outputs
        # combined should be (batch_size, src_len, hidden_dim*3)
        # combined = ???
        
        # TODO: Step 3 - Compute energy
        # energy should be (batch_size, src_len, hidden_dim)
        # energy = ???
        
        # TODO: Step 4 - Project to scalar scores
        # attention_scores should be (batch_size, src_len)
        # attention_scores = ???
        
        # TODO: Step 5 - Apply softmax
        # attention_weights should be (batch_size, src_len) and sum to 1.0
        # attention_weights = ???
        
        # TODO: Step 6 - Compute context vector
        # Use torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        # context should be (batch_size, 1, hidden_dim*2), then squeeze to (batch_size, hidden_dim*2)
        # context = ???
        
        raise NotImplementedError("TODO: Implement attention forward pass")
        
        # Once implemented, return:
        # return context, attention_weights


class DecoderWithAttention(nn.Module):
    """
    LSTM Decoder with Attention Mechanism.
    
    TODO: Implement this decoder that uses attention.
    
    Key differences from Lesson 4 decoder:
    1. Has an AttentionLayer
    2. LSTM input size is embed_dim + (hidden_dim*2) to account for context
    3. Forward pass computes attention at each step
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, padding_idx=0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        # TODO: Add attention layer
        # Hint: Use the AttentionLayer class you implemented above
        # self.attention = ???
        
        # TODO: Modify LSTM input dimension
        # The LSTM now takes embedding + attention context as input
        # Input size should be: embed_dim + (hidden_dim * 2)
        # self.lstm = nn.LSTM(???, hidden_dim, batch_first=True)
        
        # Output projection (same as Lesson 4)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        raise NotImplementedError("TODO: Initialize decoder layers")
    
    def forward(self, x, hidden, cell, encoder_outputs):
        """
        Forward pass with attention.
        
        Args:
            x: Input token (batch_size, 1)
            hidden: Hidden state (1, batch_size, hidden_dim)
            cell: Cell state (1, batch_size, hidden_dim)
            encoder_outputs: All encoder outputs (batch_size, src_len, hidden_dim*2)
            
        Returns:
            prediction: Output logits (batch_size, 1, vocab_size)
            hidden: Updated hidden state
            cell: Updated cell state
            attention_weights: Attention weights (batch_size, src_len)
        
        TODO: Implement the forward pass following these steps:
        
        Step 1: Embed the input
        
        Step 2: Compute attention context
                - Squeeze hidden from (1, batch, hidden) to (batch, hidden)
                - Call self.attention(hidden_squeezed, encoder_outputs)
        
        Step 3: Concatenate embedding with attention context
                - Unsqueeze context to match embedding shape
                - Concatenate along dim=2
        
        Step 4: Pass through LSTM
        
        Step 5: Project to vocabulary
        """
        
        # TODO: Step 1 - Embed input
        # embedded should be (batch_size, 1, embed_dim)
        # embedded = ???
        
        # TODO: Step 2 - Compute attention
        # hidden is (1, batch_size, hidden_dim), need to squeeze to (batch_size, hidden_dim)
        # context, attention_weights = ???
        
        # TODO: Step 3 - Concatenate embedding with context
        # context needs unsqueeze(1) to become (batch_size, 1, hidden_dim*2)
        # lstm_input should be (batch_size, 1, embed_dim + hidden_dim*2)
        # lstm_input = ???
        
        # TODO: Step 4 - Pass through LSTM
        # output, (hidden, cell) = ???
        
        # TODO: Step 5 - Project to vocabulary
        # prediction = ???
        
        raise NotImplementedError("TODO: Implement decoder forward pass")
        
        # Once implemented, return:
        # return prediction, hidden, cell, attention_weights


class Seq2SeqWithAttention(nn.Module):
    """
    Complete Sequence-to-Sequence Model with Attention.
    
    TODO: Implement this model that combines encoder and attention decoder.
    
    Key differences from Lesson 4:
    1. Encoder returns ALL outputs (not just final state)
    2. Decoder receives all encoder outputs for attention
    3. We collect attention weights for visualization
    """
    
    def __init__(self, encoder, attention_decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = attention_decoder
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass with attention.
        
        Args:
            src: Source sequence (batch_size, src_len)
            trg: Target sequence (batch_size, trg_len)
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Predictions (batch_size, trg_len, vocab_size)
            attentions: Attention weights (batch_size, trg_len, src_len)
        
        TODO: Implement the forward pass following these steps:
        
        Step 1: Encode source sequence
                - Get embeddings from encoder
                - Get ALL encoder outputs (not just final state)
                - Get final hidden and cell states
        
        Step 2: Initialize output tensors
                - outputs for predictions
                - attentions for attention weights
        
        Step 3: Decode with attention
                - For each target position:
                  - Call decoder with encoder_outputs
                  - Collect attention weights
                  - Apply teacher forcing
        """
        
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc.out_features
        src_len = src.shape[1]
        
        # TODO: Step 1 - Encode source
        # Get encoder embeddings and outputs
        # embedded = self.encoder.embedding(src)
        # encoder_outputs, (hidden, cell) = self.encoder.lstm(embedded)
        # 
        # Combine bidirectional states for hidden and cell
        # hidden = torch.tanh(self.encoder.fc_hidden(...)).unsqueeze(0)
        # cell = torch.tanh(self.encoder.fc_cell(...)).unsqueeze(0)
        
        # TODO: Step 2 - Initialize output tensors
        # outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        # attentions = torch.zeros(batch_size, trg_len, src_len).to(src.device)
        
        # TODO: Step 3 - Decode with attention
        # decoder_input = trg[:, 0].unsqueeze(1)
        # 
        # for t in range(trg_len):
        #     # Call decoder with encoder_outputs for attention
        #     output, hidden, cell, attention_weights = self.decoder(
        #         decoder_input, hidden, cell, encoder_outputs
        #     )
        #     
        #     outputs[:, t:t+1, :] = output
        #     attentions[:, t, :] = attention_weights
        #     
        #     # Teacher forcing logic
        #     if t < trg_len - 1:
        #         use_teacher_forcing = random.random() < teacher_forcing_ratio
        #         decoder_input = trg[:, t+1].unsqueeze(1) if use_teacher_forcing else output.argmax(2)
        
        raise NotImplementedError("TODO: Implement seq2seq forward pass")
        
        # Once implemented, return:
        # return outputs, attentions


# ============================================================================
# Utility Functions (PROVIDED)
# ============================================================================

def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing Models (STARTER VERSION)")
    print("="*80)
    print("\n⚠️  You need to implement the attention classes first!")
    print("   Complete the TODOs in:")
    print("   - AttentionLayer")
    print("   - DecoderWithAttention")
    print("   - Seq2SeqWithAttention")
    print("\n   Then run the tests in the notebook!")
