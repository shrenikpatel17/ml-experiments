import argparse, torch, numpy as np
from transformers import AutoTokenizer, AutoModel

#CLI
p = argparse.ArgumentParser()
p.add_argument("--model", default="gpt2")
p.add_argument("--topk", type=int, default=15)
p.add_argument("--terms", nargs="*", default=[
    "bioweapon", "anthrax", "malware", "payload",
    "exploit", "ransomware", "virus", "toxin"
])
args = p.parse_args()

tok  = AutoTokenizer.from_pretrained(args.model)
tok.pad_token = tok.eos_token
mdl  = AutoModel.from_pretrained(args.model).eval()

with torch.no_grad():
    W = mdl.get_input_embeddings().weight            
    Wn = W / W.norm(dim=1, keepdim=True)

vocab = np.array(tok.convert_ids_to_tokens(range(len(tok))))

def mean_embed(word):
    ids = tok.encode(" "+word, add_special_tokens=False)
    vec = Wn[ids].mean(0)
    return vec / vec.norm()

def is_word(tok_str):
    return tok_str.startswith("Ġ") and tok_str[1:].isalpha() and len(tok_str) > 3

def nearest(vec, k):
    sim = (Wn @ vec).cpu().numpy()
    order = np.argsort(sim)[::-1]
    result = []
    for idx in order:
        t = vocab[idx]
        if is_word(t):
            result.append(t[1:])      
            if len(result) == k:
                break
    return result

for w in args.terms:
    vec = mean_embed(w)
    neigh = nearest(vec, args.topk)
    print(f"\n🔍 Nearest {args.topk} words to “{w}”:")
    print("   " + ", ".join(neigh))