import torch
import numpy as np
from copy import deepcopy
from gpt_download import download_and_load_gpt2
from model import GPTModel
from config import GPT_CONFIG_SMALL, GPT_CONFIG_MEDIUM, GPT_CONFIG_LARGE, GPT_CONFIG_XLARGE

CONFIGS = {
    'small': GPT_CONFIG_SMALL,
    'medium': GPT_CONFIG_MEDIUM,
    'large': GPT_CONFIG_LARGE,
    'xlarge': GPT_CONFIG_XLARGE
}

GPT2_SIZE_MAP = {
    'small': '124M',
    'medium': '355M',
    'large': '774M',
    'xlarge': '1558M'
}


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. Left: {left.shape}, Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right, device=left.device))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )
        
        q_b, k_b, v_b = np.split(
            params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b
        )
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )
        
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )
        
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )
        
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"]
        )
        
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"]
        )
    
    gpt.final_layer.scale = assign(gpt.final_layer.scale, params["g"])
    gpt.final_layer.shift = assign(gpt.final_layer.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def load_weights_into_classifier(classifier, params, num_classes=2):
    classifier.pos_emb.weight = assign(classifier.pos_emb.weight, params['wpe'])
    classifier.tok_emb.weight = assign(classifier.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1
        )
        classifier.trf_blocks[b].att.W_query.weight = assign(
            classifier.trf_blocks[b].att.W_query.weight, q_w.T
        )
        classifier.trf_blocks[b].att.W_key.weight = assign(
            classifier.trf_blocks[b].att.W_key.weight, k_w.T
        )
        classifier.trf_blocks[b].att.W_value.weight = assign(
            classifier.trf_blocks[b].att.W_value.weight, v_w.T
        )
        
        q_b, k_b, v_b = np.split(
            params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1
        )
        classifier.trf_blocks[b].att.W_query.bias = assign(
            classifier.trf_blocks[b].att.W_query.bias, q_b
        )
        classifier.trf_blocks[b].att.W_key.bias = assign(
            classifier.trf_blocks[b].att.W_key.bias, k_b
        )
        classifier.trf_blocks[b].att.W_value.bias = assign(
            classifier.trf_blocks[b].att.W_value.bias, v_b
        )
        
        classifier.trf_blocks[b].att.out_proj.weight = assign(
            classifier.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        classifier.trf_blocks[b].att.out_proj.bias = assign(
            classifier.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )
        
        classifier.trf_blocks[b].ff.layers[0].weight = assign(
            classifier.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        classifier.trf_blocks[b].ff.layers[0].bias = assign(
            classifier.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        
        classifier.trf_blocks[b].ff.layers[2].weight = assign(
            classifier.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        classifier.trf_blocks[b].ff.layers[2].bias = assign(
            classifier.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )
        
        classifier.trf_blocks[b].norm1.scale = assign(
            classifier.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"]
        )
        classifier.trf_blocks[b].norm1.shift = assign(
            classifier.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"]
        )
        
        classifier.trf_blocks[b].norm2.scale = assign(
            classifier.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"]
        )
        classifier.trf_blocks[b].norm2.shift = assign(
            classifier.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"]
        )
    
    classifier.final_layer.scale = assign(classifier.final_layer.scale, params["g"])
    classifier.final_layer.shift = assign(classifier.final_layer.shift, params["b"])

def load_pretrained_gpt2(model_size='small', models_dir='gpt2', device='cpu'):   
    config = deepcopy(CONFIGS[model_size])
    config['qkv_bias'] = True
    
    print(f"Downloading/loading GPT-2 {GPT2_SIZE_MAP[model_size]}...")
    settings, params = download_and_load_gpt2(
        model_size=GPT2_SIZE_MAP[model_size],
        models_dir=models_dir
    )
    
    model = GPTModel(config).to(device)
    load_weights_into_gpt(model, params)
    model.eval()
    
    print(f"✓ Loaded pretrained GPT-2 {model_size}")
    return model, config

def load_classifier(model_path, device='cpu', num_classes=2):
    from copy import deepcopy
    from model import GPTClassifier
    from config import GPT_CONFIG_SMALL
    
    print(f"Loading classifier from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'config' in checkpoint and checkpoint['config'] is not None:
        config = checkpoint['config']
        print("✓ Config loaded from checkpoint")
    else:
        config = deepcopy(GPT_CONFIG_SMALL)
        config['qkv_bias'] = True
        print("⚠ Using default config")
    
    model = GPTClassifier(config, num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Classifier loaded!")
    return model, config