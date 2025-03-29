import sys
import os
import json  

import torch  

import tiktoken  
from tiktoken.load import load_tiktoken_bpe 

from tokenizer import Tokenizer

# References:
# https://github.com/therealoliver/Deepdive-llama3-from-scratch?tab=readme-ov-file
# https://gist.github.com/kevmo314/294001659324429bae6749062a9003db

base_path = '/opt/hive/foundation_models/llama1b/Llama3.2-1B/'

def rms_norm(tensor, norm_weights):
    """
    Define the calculation function for RMS normalization. Each token will be normalized independently.
    
    norm_weights is the pre-trained scaling factor (i.e., gi in the formula) to enhance the model's representational ability. It can be loaded from the model file and has 2048 dimensions

    torch.rsqrt used to calculates the reciprocal of the square root of a tensor, i.e., 1/RMS(a)
    """
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights

def get_freqs_cis(dim, num_tokens):
    """
    get frequencies for RoPE
    """
    # [64] -> [32]
    dim_after_splitting_into_pairs = dim//2
    # Obtain i/D. 32 theta values are required
    zero_to_one_split_into_32_parts = torch.tensor(range(dim_after_splitting_into_pairs))/dim_after_splitting_into_pairs
    # Compute thetas. 
    # [32]
    freqs = 1.0 / (rope_theta ** zero_to_one_split_into_32_parts)
    # Calculate m*theta. Need frequencies for each token
    # [N] & [32] -> [Nx32]
    freqs_for_each_token = torch.outer(torch.arange(num_tokens), freqs)
    # convert m*theta to complex form (first arg is 1, e.g. modulus 1, and second arg is angle) 
    freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)  # [17x64] -> [17x64]

    return freqs_cis

def apply_RoPE(q_per_token):
    """
    Applys rotation to the Q/K vectors. Here everything says q, e.g. q_per_token, but applies the same to k
    """

    # Calculate freqs_cis
    freqs_cis = get_freqs_cis(int(q_per_token.shape[1]), q_per_token.shape[0])

    # Divide vector into pairs along the dimensions direction to form dimension pairs. [Nx64] -> [Nx32x2]
    q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2) 
    # Convert to complex number representation, (x,y) -> (x+yi). [Nx32x2] -> [Nx32]
    q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
    # Calculate (x+yi)*(cosmθ+sinmθi) to complete the rotation operation. [Nx32] * [Nx32] = [Nx32]  
    q_per_token_as_complex_numbers_rotated = q_per_token_as_complex_numbers * freqs_cis  
    # Convert the result back to real number representation, (x+yi) -> (x,y). [Nx32] -> [Nx32x2]
    q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers_rotated)  
    # Convert the result back to the original vector shape to obtain the final query vector. [Nx32x2] -> [Nx64]
    q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)  

    return q_per_token_rotated

def load_model_and_tokenizer():

    print("Loading model...")
    model = torch.load(
        os.path.join(base_path, 'consolidated.00.pth'), 
        map_location=torch.device('cpu'),
        mmap=False, 
        weights_only=True)
    print("Model Loaded!")
    # First 20 keys
    print(json.dumps(list(model.keys())[:20], indent=4))

    tokenizer_path = os.path.join(base_path, 'tokenizer.model')
    tokenizer = Tokenizer(model_path=tokenizer_path)

    return model, tokenizer

with open(os.path.join(base_path, 'params.json'), "r") as f:
    config = json.load(f)
print(json.dumps(config, indent=4))

# Record these configurations, which will be gradually used later.
dim = config["dim"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
vocab_size = config["vocab_size"]
multiple_of = config["multiple_of"]
ffn_dim_multiplier = config["ffn_dim_multiplier"]
norm_eps = config["norm_eps"]
rope_theta = torch.tensor(config["rope_theta"])


def forward(model, tokenizer, input_string):
    """
    Run one forward pass of them model. Returns both the next output (appended to the initial string) 
    and the output logits
    """
    
    # Initial encoding of input
    tokens = tokenizer.encode(input_string, bos=True, eos=False)
    tokens = torch.tensor(tokens)

    # Lookup table layer, shape is torch.Size([128256, 2048]) 
    embedding_layer = torch.nn.Embedding(vocab_size, dim)
    embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
    # Converts tokens to embedding space via lookup, no token interaction
    # [N] -> [Nx2048]
    token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)
    # Use the embeddings of the input tokens as the initial input.
    final_embedding = token_embeddings_unnormalized  

    # Perform layer-by-layer calculation for the 32-layer Transformer blocks
    for layer in range(n_layers):
        
        # The first normalization
        # [Nx2048] & [2048] -> [Nx2048]
        layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])  
        
        # The first feature transformation - Multi-Head Self-Attention
        # Obtain the qkv weight matrix of the attention mechanism for the current layer
        q_layer = model[f"layers.{layer}.attention.wq.weight"]  # [2048x2048]
        q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, dim)  # [32x64x2048]
        k_layer = model[f"layers.{layer}.attention.wk.weight"]  # [512x2048]
        k_layer = k_layer.view(n_kv_heads, k_layer.shape[0] // n_kv_heads, dim)  # [8x64x2048]
        v_layer = model[f"layers.{layer}.attention.wv.weight"]  # [512x2048]
        v_layer = v_layer.view(n_kv_heads, v_layer.shape[0] // n_kv_heads, dim)  # [8x64x2048]
        
        # Used to store the calculation results of the attention mechanism for each head
        qkv_attention_store = []
        
        # Calculate the attention mechanism results for each head
        for head in range(n_heads):
            # Extract the QKV weight matrices corresponding to the current head
            q_layer_head = q_layer[head]  # [32x64x2048] -> [64x2048]
            k_layer_head = k_layer[head//4]  # Every 4 heads share one key weight, [8x64x2048] -> [64x2048]
            v_layer_head = v_layer[head//4]  # Every 4 heads share one value weight, [8x64x2048] -> [64x2048]
            
            # Calculate XW to obtain the QKV vectors
            # [Nx2048] x [2048x64] = [Nx64]
            q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
            k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
            v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)
            
            # Apply RoPE to Q and K vectors
            q_per_token_rotated = apply_RoPE(q_per_token)
            k_per_token_rotated = apply_RoPE(k_per_token)

            # Calculate the attention scores and normalize the scores simultaneously (i.e., Q×K/sqrt(dim))
            # [Nx64] x [64xN] = [NxN]
            qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(q_per_token_rotated.shape[1])**0.5  
            
            # Mask the scores of future tokens, create matrix of -inf and shape [NxN]
            mask = torch.full(qk_per_token.shape, float("-inf"), device=qk_per_token.device)  
            # Keep the negative infinity in the upper-triangular part and set others to 0 
            # (i.e., the upper-triangular area represents future tokens that need to be masked). 
            # The diagonal offset is 1 to avoid masking the token itself. [NxN]
            mask = torch.triu(mask, diagonal=1)  
            # Add attention scoress to masking matrix, -infs will go to zero after softmax [NxN]
            qk_per_token_after_masking = qk_per_token + mask
            
            # Calculate the attention weights (i.e., softmax(score))
            # Meanwhile, convert it back to half-precision (because it will be multiplied with the value vector v_per_token later, so the data types need to be the same).
            # Calculate the softmax row-by-row. [NxN]
            qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)  
            
            # Calculate the final result of the attention mechanism (i.e., softmax(score) × V)
            qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)  # [NxN] x [Nx64] = [Nx64]
            
            # Record the result of this head
            qkv_attention_store.append(qkv_attention)
        
        # Merge the multi-head attention results
        # Merge the second dimension, that is, 32x[Nx64] -> [Nx2048]
        stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
        
        # Perform a linear mapping on the results to generate the final multi-head self-attention mechanism results
        o_layer = model[f"layers.{layer}.attention.wo.weight"]
        embedding_delta = torch.matmul(stacked_qkv_attention, o_layer.T)  # [Nx2048] x [2048x2048] = [Nx2048]


        # The first Residual Operation
        # Add the output of the attention layer to the original input to complete the residual operation
        embedding_after_edit = final_embedding + embedding_delta  # [Nx2048] + [Nx2048] = [Nx2048]
        
        # The second normalization
        # [Nx2048] & [2048] -> [Nx2048]
        embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])  
        
        # The second feature transformation - Feed-Forward Network 
        
        # Load the parameter matrix of the feed-forward network (SwiGLU)
        w1 = model[f"layers.{layer}.feed_forward.w1.weight"]  # [8192x2048]
        w3 = model[f"layers.{layer}.feed_forward.w3.weight"]  # [8192x2048]
        w2 = model[f"layers.{layer}.feed_forward.w2.weight"]  # [2048x8192]
        
        # Calculate the results of the feed-forward network (output = (silu(XW1) * XW3)W2)
        # [Nx2048] x [2048x8192] x [8192x2048] = [Nx2048]
        output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
        
        # The second residual operation, obtain the final output result of the current Transformer block
        # Add the output of the feed-forward layer to the original input to complete the residual operation
        final_embedding = embedding_after_edit+output_after_feedforward  # [Nx2048] + [Nx2048] = [Nx2048]

    final_embedding = rms_norm(final_embedding, model["norm.weight"])
    logits = torch.matmul(final_embedding[-1], model["output.weight"].T)  # [Nx2048] -> [2048] -> [2048] x [2048x128256] = [128256]
    next_token = torch.argmax(logits, dim=-1)  
    final_string = tokenizer.decode([next_token.item()])
    input_string = input_string + final_string

    return input_string, logits

if __name__ == "__main__":

    model, tokenizer = load_model_and_tokenizer()

    input_string = sys.argv[1]
    print(input_string)
    for _ in range(10):
        input_string, logits = forward(model, tokenizer, input_string)
        print(input_string)
