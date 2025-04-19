import argparse, random, csv, numpy as np, torch, matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

#Configuration
hazard = ["bioweapon","anthrax","malware","payload",
          "exploit","ransomware","virus","toxin"]
parser = argparse.ArgumentParser()
parser.add_argument("--modelA", default="gpt2")
parser.add_argument("--modelB", default="cais/Zephyr_RMU")
parser.add_argument("--topk",  type=int, default=15)
args = parser.parse_args()
models = [args.modelA, args.modelB]

DEVICE = "cpu"

#Helpers
def load(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    mdl = AutoModel.from_pretrained(model_name).eval()
    with torch.no_grad():
        W = mdl.get_input_embeddings().weight.to(DEVICE)  # [V,d]
        Wn = W / W.norm(dim=1, keepdim=True)
    vocab = np.array(tok.convert_ids_to_tokens(range(len(tok))))
    return tok, Wn, vocab

def mean_vec(word, tok, Wn):
    ids = tok.encode(" "+word, add_special_tokens=False)
    vec = Wn[ids].mean(0)
    return vec / vec.norm()

def ok(t):                      
    return t.startswith("Ġ") and t[1:].isalpha() and len(t) > 3

def neighbours(vec, Wn, vocab, k):
    sim = (Wn @ vec).cpu().numpy()
    order = sim.argsort()[::-1]
    out=[]
    for idx in order:
        if ok(vocab[idx]):
            out.append((vocab[idx][1:], sim[idx]))  
            if len(out)==k: break
    return out, sim

#Work per model
records = {}
sims_for_hist = {}
for m in models:
    print(f"→ loading {m}")
    tok,Wn,vocab = load(m)
    rec, sims = {}, []
    for w in hazard:
        v = mean_vec(w, tok, Wn)
        nbs, all_sim = neighbours(v, Wn, vocab, args.topk)
        rec[w] = nbs
        sims.append(all_sim)
    records[m] = rec
    sims_for_hist[m] = sims      

    with open(f"neighbours_{m.split('/')[-1]}.csv","w",newline='') as f:
        wr = csv.writer(f); wr.writerow(["probe","neighbour","cosine"])
        for p, lst in rec.items():
            for nb,cs in lst: wr.writerow([p,nb,f"{cs:.4f}"])

# Figure 1 – cosine histograms
plt.figure(figsize=(7,3.2))
for i,m in enumerate(models):
    sims = np.concatenate([s for s in sims_for_hist[m]]) 
    rand_idx = random.sample(range(len(sims)), 300)
    rand = sims[rand_idx]
    probe = sims[:len(sims)//len(hazard)]  
    plt.subplot(1,2,i+1)
    plt.hist(rand, bins=30, alpha=.6, label="random vocab")
    plt.hist(probe, bins=30, alpha=.6, label="hazard probes")
    plt.title(m.split('/')[-1]); plt.xlabel("cosine similarity"); plt.ylim(0,1200)
    plt.axvline(0,color="k"); plt.legend()
plt.tight_layout(); plt.savefig("fig_histograms.png"); plt.savefig("fig_histograms.pdf")
print("Saved fig_histograms.(png|pdf)")

# Figure 2 – neighbour‑set overlap
import itertools
k = args.topk
overlap = np.zeros((len(hazard),1))
for i,p in enumerate(hazard):
    setA = {w for w,_ in records[models[0]][p]}
    setB = {w for w,_ in records[models[1]][p]}
    overlap[i,0] = 100*len(setA & setB)/k

plt.figure(figsize=(3,4))
plt.imshow(overlap, cmap="Blues", vmin=0, vmax=100)
plt.colorbar(label="Jaccard % (top‑15)")
plt.yticks(range(len(hazard)), hazard); plt.xticks([0],[f"{k} NN\nOverlap"])
plt.title("Embedding‑Neighbour Overlap")
for y,x in itertools.product(range(len(hazard)),[0]):
    plt.text(x, y, f"{overlap[y,x]:.0f}%", ha="center", va="center", color="black")
plt.tight_layout(); plt.savefig("fig_overlap.png"); plt.savefig("fig_overlap.pdf")
print("Saved fig_overlap.(png|pdf)")