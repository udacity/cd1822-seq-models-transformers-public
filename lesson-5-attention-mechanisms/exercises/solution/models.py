"""
models.py - Seq2Seq Models with Attention (SOLUTION VERSION)

This module contains:
- Encoder (from Lesson 4 - PROVIDED)
- AttentionLayer (COMPLETE SOLUTION)
- DecoderWithAttention (COMPLETE SOLUTION)
- Seq2SeqWithAttention (COMPLETE SOLUTION)
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
# SOLUTION FOR LESSON 5 - Complete implementations
# ============================================================================

class AttentionLayer(nn.Module):
    """
    Bahdanau Attention Mechanism (COMPLETE SOLUTION).
    
    Computes attention weights and context vectors.
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        # Attention network layers
        # Input: [hidden_dim (decoder) + hidden_dim*2 (encoder)] = hidden_dim*3
        # Output: hidden_dim
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # Projection to scalar scores
        # Input: hidden_dim
        # Output: 1 (scalar score)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        """
        Compute attention weights and context vector.
        
        Args:
            hidden: Decoder hidden state (batch_size, hidden_dim)
            encoder_outputs: All encoder hidden states (batch_size, src_len, hidden_dim*2)
            
        Returns:
            context: Attention context vector (batch_size, hidden_dim*2)
            attention_weights: Attention distribution (batch_size, src_len)
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Step 1: Repeat decoder hidden state for all source positions
        # From (batch_size, hidden_dim) to (batch_size, src_len, hidden_dim)
        hidden_repeated = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Step 2: Concatenate with encoder outputs
        # Result: (batch_size, src_len, hidden_dim*3)
        combined = torch.cat((hidden_repeated, encoder_outputs), dim=2)
        
        # Step 3: Compute energy (attention scores)
        # Apply: tanh(W @ combined)
        # Result: (batch_size, src_len, hidden_dim)
        energy = torch.tanh(self.attn(combined))
        
        # Step 4: Project to scalar scores
        # Apply: v @ energy
        # Result: (batch_size, src_len, 1) -> squeeze to (batch_size, src_len)
        attention_scores = self.v(energy).squeeze(2)
        
        # Step 5: Apply softmax to get attention weights
        # Result: (batch_size, src_len) with values summing to 1.0
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Step 6: Compute context vector as weighted sum
        # attention_weights: (batch_size, src_len)
        # encoder_outputs: (batch_size, src_len, hidden_dim*2)
        # Use bmm with unsqueeze: (batch_size, 1, src_len) @ (batch_size, src_len, hidden_dim*2)
        # Result: (batch_size, 1, hidden_dim*2) -> squeeze to (batch_size, hidden_dim*2)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights


class DecoderWithAttention(nn.Module):
    """
    LSTM Decoder with Attention Mechanism (COMPLETE SOLUTION).
    
    Integrates attention into the decoder to access all encoder states.
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, padding_idx=0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dim)
        
        # LSTM input is now: embedding + attention context
        # embed_dim + (hidden_dim * 2 from bidirectional encoder)
        self.lstm = nn.LSTM(embed_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        
        # Output projection (same as Lesson 4)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
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
        """
        # Step 1: Embed the input
        # Result: (batch_size, 1, embed_dim)
        embedded = self.embedding(x)
        
        # Step 2: Compute attention context
        # hidden is (1, batch_size, hidden_dim), need to squeeze to (batch_size, hidden_dim)
        hidden_squeezed = hidden.squeeze(0)
        context, attention_weights = self.attention(hidden_squeezed, encoder_outputs)
        
        # Step 3: Concatenate embedding with attention context
        # context is (batch_size, hidden_dim*2), need to unsqueeze to (batch_size, 1, hidden_dim*2)
        context = context.unsqueeze(1)
        # Result: (batch_size, 1, embed_dim + hidden_dim*2)
        lstm_input = torch.cat((embedded, context), dim=2)
        
        # Step 4: Pass through LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Step 5: Project to vocabulary
        prediction = self.fc(output)
        
        return prediction, hidden, cell, attention_weights


class Seq2SeqWithAttention(nn.Module):
    """
    Complete Sequence-to-Sequence Model with Attention (COMPLETE SOLUTION).
    
    Combines encoder with attention-enhanced decoder.
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
        """
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc.out_features
        src_len = src.shape[1]
        
        # Step 1: Encode source sequence
        # Get embeddings and ALL encoder outputs (not just final state)
        embedded = self.encoder.embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder.lstm(embedded)
        
        # Combine bidirectional states for hidden and cell
        hidden = torch.tanh(
            self.encoder.fc_hidden(torch.cat([hidden[-2], hidden[-1]], dim=1))
        ).unsqueeze(0)
        
        cell = torch.tanh(
            self.encoder.fc_cell(torch.cat([cell[-2], cell[-1]], dim=1))
        ).unsqueeze(0)
        
        # Step 2: Initialize output tensors
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        attentions = torch.zeros(batch_size, trg_len, src_len).to(src.device)
        
        # Step 3: Decode with attention
        # First input is <SOS> token
        decoder_input = trg[:, 0].unsqueeze(1)
        
        for t in range(trg_len):
            # Call decoder with encoder_outputs for attention
            output, hidden, cell, attention_weights = self.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
            
            # Store output and attention weights
            outputs[:, t:t+1, :] = output
            attentions[:, t, :] = attention_weights
            
            # Decide whether to use teacher forcing
            if t < trg_len - 1:
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                
                if use_teacher_forcing:
                    # Use actual next token from target
                    decoder_input = trg[:, t+1].unsqueeze(1)
                else:
                    # Use model's prediction
                    decoder_input = output.argmax(2)
        
        return outputs, attentions


# ============================================================================
# Utility Functions (PROVIDED)
# ============================================================================

def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing Models (SOLUTION VERSION)")
    print("="*80)
    print("\n✓ All attention classes are fully implemented!")
    print("   - AttentionLayer")
    print("   - DecoderWithAttention")
    print("   - Seq2SeqWithAttention")
    print("\n✓ Ready to run in notebook!")
