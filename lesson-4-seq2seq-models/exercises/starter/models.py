"""
Encoder-Decoder Q&A Model Architecture.
"""
import random
import torch
import torch.nn as nn

from data import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_TOKEN


class Encoder(nn.Module):
    """
    Bidirectional LSTM Encoder.
    
    Processes (context + SEP + question) and compresses information into
    a fixed-size context vector - THE BOTTLENECK!
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, padding_idx=PAD_TOKEN):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # TODO: Create embedding layer
        # Use nn.Embedding with padding_idx parameter
        
        # TODO: Create bidirectional LSTM
        # Set batch_first=True and bidirectional=True
        
        # TODO: Create linear layers to combine forward/backward states
        # Need to project 2*hidden_dim → hidden_dim for both hidden and cell states
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len) with token indices
        
        Returns:
            hidden: Final context vector (1, batch_size, hidden_dim)
            cell: Final cell state (1, batch_size, hidden_dim)
        """
        # TODO: Embed input tokens
        
        # TODO: Pass through bidirectional LSTM
        # Note: Output is (outputs, (hidden, cell))
        # hidden shape: (2, batch_size, hidden_dim) because bidirectional
        
        # TODO: Combine forward and backward hidden/cell states
        # Forward state: hidden[-2], Backward state: hidden[-1]
        # Use linear layers and tanh activation
        # Return with unsqueeze(0) to get (1, batch_size, hidden_dim)
        pass


class Decoder(nn.Module):
    """
    LSTM Decoder.
    
    Generates answer tokens autoregressively using hidden state from encoder
    (the compressed context vector).
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, padding_idx=PAD_TOKEN):
        super().__init__()
        
        # TODO: Create embedding layer
        # Use nn.Embedding with padding_idx parameter
        
        # TODO: Create LSTM (unidirectional)
        # batch_first=True, no bidirectional
        
        # TODO: Create output projection layer
        # Project hidden_dim → vocab_size for token predictions
    
    def forward(self, x, hidden, cell):
        """
        Args:
            x: Input tensor (batch_size, seq_len) - currently generated tokens
            hidden: Hidden state from encoder or previous decoder step
            cell: Cell state from encoder or previous decoder step
        
        Returns:
            prediction: Token logits (batch_size, seq_len, vocab_size)
            hidden: Updated hidden state
            cell: Updated cell state
        """
        # TODO: Embed input tokens
        
        # TODO: Pass through LSTM with previous hidden/cell states
        
        # TODO: Project output to vocabulary size for token predictions
        pass


class Seq2Seq(nn.Module):
    """
    Complete Seq2Seq Model for Q&A.
    
    Combines encoder and decoder with teacher forcing during training.
    """
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: Source tokens (batch_size, src_len) - context + SEP + question
            trg: Target tokens (batch_size, trg_len) - answer
            teacher_forcing_ratio: Probability of using ground truth tokens vs predictions
        
        Returns:
            outputs: Predictions (batch_size, trg_len, vocab_size)
        """
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc.out_features
        
        # TODO: Initialize output tensor for predictions
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        
        # TODO: Encode source
        # This creates the BOTTLENECK: all context → single vector
        hidden, cell = self.encoder(src)
        
        # TODO: Start with SOS token (first token of target)
        decoder_input = trg[:, 0].unsqueeze(1)
        
        # TODO: Decode autoregressively
        # For each timestep:
        #   1. Pass decoder_input through decoder
        #   2. Store predictions
        #   3. Decide on next input: ground truth (teacher forcing) or prediction
        for t in range(trg_len):
            # TODO: Get decoder output
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t:t+1, :] = output
            
            if t < trg_len - 1:
                # TODO: Implement teacher forcing decision
                # With probability teacher_forcing_ratio, use ground truth
                # Otherwise use model prediction (argmax)
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                decoder_input = trg[:, t+1].unsqueeze(1) if use_teacher_forcing else output.argmax(2)
        
        return outputs


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
