import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import random
import os

# --- Configuration ---
DATASET_NAME = "kchawla123/casino"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
LEARNING_RATE = 0.005
EPOCHS = 1000
SEED = 42
BETA = 0.2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

# --- 1. Data Processing ---
PREF_MAP = {'Low': 0.0, 'Medium': 0.5, 'High': 1.0}

def heuristic_claim_parser(text):
    text = text.lower()
    high_markers = ["need", "want", "have to", "vital", "my", "give me", "i get"]
    low_markers = ["you take", "you have", "yours", "don't need", "useless", "give you"]
    score = 0.5 
    if any(m in text for m in high_markers): score = 1.0
    elif any(m in text for m in low_markers): score = 0.0
    return score

def parse_casino_data_with_history():
    print(f"--- Downloading {DATASET_NAME} ---")
    dataset = load_dataset(DATASET_NAME, split='train')
    
    samples = []
    
    for chat in dataset:
        p_info = chat['participant_info']
        truth_lookup = {}
        for agent_key in ['mturk_agent_1', 'mturk_agent_2']:
            v2i = p_info[agent_key]['value2issue']
            for priority, item_name in v2i.items():
                if item_name: truth_lookup[(agent_key, item_name.lower())] = PREF_MAP[priority]

        local_history = {'mturk_agent_1': None, 'mturk_agent_2': None}

        for turn in chat['chat_logs']:
            text = turn['text']
            speaker_id = turn['id'] 
            if not text or not speaker_id: continue

            items_mentioned = [i for i in ['food', 'water', 'firewood'] if i in text.lower()]
            if len(items_mentioned) != 1: continue 
            target_item = items_mentioned[0]
            
            if (speaker_id, target_item) not in truth_lookup: continue
            
            true_pref = truth_lookup[(speaker_id, target_item)]
            claimed_pref = heuristic_claim_parser(text)
            
            prev_data = local_history[speaker_id]
            prev_true = prev_data['true_pref'] if prev_data else 0.5
            prev_claim = prev_data['claim'] if prev_data else 0.5
            
            samples.append({
                'text': text,
                'curr_true': true_pref,      
                'curr_claim': claimed_pref, 
                'prev_true': prev_true,
                'prev_claim': prev_claim
            })
            local_history[speaker_id] = {'true_pref': true_pref, 'claim': claimed_pref}
            
    df = pd.DataFrame(samples)
    print(f"--- Processed {len(df)} samples with Temporal History ---")
    return df

# --- 2. The Hybrid MLNN ---
class TrustHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )
    def forward(self, embeddings):
        return self.net(embeddings).squeeze()

class TemporalMLNN(nn.Module):
    def __init__(self, embedding_dim, tau=0.1):
        super().__init__()
        self.trust_head = TrustHead(embedding_dim)
        self.tau = tau 
        self.prior_belief = 0.5
    
    def softmin(self, values):
        neg_x = -values / self.tau
        return -self.tau * torch.logsumexp(neg_x, dim=1)

    def forward(self, embeddings, curr_claim, prev_claim, curr_truth, prev_truth):
        trust_scores = self.trust_head(embeddings)
        pred_belief = trust_scores * curr_claim + (1.0 - trust_scores) * self.prior_belief
        
        world_curr_truth = 1.0 - torch.abs(curr_claim - curr_truth)
        world_prev_truth = 1.0 - torch.abs(prev_claim - prev_truth)
        
        imp_curr = torch.clamp(1.0 - trust_scores + world_curr_truth, 0.0, 1.0)
        imp_prev = torch.clamp(1.0 - trust_scores + world_prev_truth, 0.0, 1.0)
        
        stack = torch.stack([imp_curr, imp_prev], dim=1)
        consistency = self.softmin(stack)
        
        return pred_belief, consistency, trust_scores

# --- 3. Visualization & Output Functions ---
def save_trust_plot_pdf(df, filename="casino_hybrid_results.pdf"):
    print(f"\n[Plotting] Generating {filename} with figsize=(6,2)...")
    sns.set_theme(style="whitegrid", font_scale=0.8)
    plt.figure(figsize=(6, 2))
    
    sns.kdeplot(data=df[df['Honest']], x='Learned_Trust', fill=True, color='#2ecc71', label='Consistent Honest', alpha=0.3, linewidth=1.2, cut=0)
    sns.kdeplot(data=df[df['Reformed']], x='Learned_Trust', fill=True, color='#f39c12', label='Reformed Liar', alpha=0.4, linewidth=1.2, cut=0)
    sns.kdeplot(data=df[df['Liar']], x='Learned_Trust', fill=True, color='#e74c3c', label='Current Liar', alpha=0.3, linewidth=1.2, cut=0)
    
    plt.xlim(0, 1.0)
    plt.xlabel("Learned Trust ($A_{\\theta}$)", fontweight='bold')
    plt.ylabel("Density")
    plt.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.25), fontsize=8)
    sns.despine(left=True)
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"[Plotting] Saved to {filename}")

def print_qualitative_examples(df):
    """Prints random examples from each category to stdout."""
    print("\n" + "="*60)
    print(f"{'QUALITATIVE EXAMPLES':^60}")
    print("="*60)
    
    categories = {
        "Consistent Honest": df['Honest'],
        "Current Liar": df['Liar'],
        "Reformed Liar": df['Reformed']
    }

    for label, mask in categories.items():
        print(f"\n--- {label} ---")
        subset = df[mask]
        
        if subset.empty:
            print("(No examples found)")
            continue
            
        # Sample 3 random examples
        n_samples = min(3, len(subset))
        sample = subset.sample(n=n_samples, random_state=SEED)
        
        for idx, row in sample.iterrows():
            clean_text = row['text'].replace('\n', ' ').strip()
            # Truncate if too long for display
            if len(clean_text) > 100: clean_text = clean_text[:97] + "..."
            
            print(f"[Trust: {row['Learned_Trust']:.4f}] \"{clean_text}\"")

# --- 4. Main Execution ---
def run_hybrid_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Hybrid Experiment on {device} ---")

    df = parse_casino_data_with_history()
    if df.empty: return

    print("Embedding text...")
    encoder = SentenceTransformer(EMBEDDING_MODEL, device=str(device))
    embeddings = encoder.encode(df['text'].tolist(), convert_to_tensor=True).to(device).clone()
    
    curr_c = torch.tensor(df['curr_claim'].values, dtype=torch.float32).to(device)
    prev_c = torch.tensor(df['prev_claim'].values, dtype=torch.float32).to(device)
    curr_t = torch.tensor(df['curr_true'].values, dtype=torch.float32).to(device)
    prev_t = torch.tensor(df['prev_true'].values, dtype=torch.float32).to(device)
    
    model = TemporalMLNN(EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n--- Training Hybrid Model (Task + Logic) ---")
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        pred_belief, consistency, trust = model(embeddings, curr_c, prev_c, curr_t, prev_t)
        
        task_loss = torch.mean((pred_belief - curr_t) ** 2)
        logic_loss = torch.mean(1.0 - consistency)
        loss = task_loss + BETA * logic_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Total {loss.item():.4f}")

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        _, _, final_trust = model(embeddings, curr_c, prev_c, curr_t, prev_t)
        df['Learned_Trust'] = final_trust.cpu().numpy()

    # Define Groups
    df['Honest'] = (df['curr_claim'] == df['curr_true']) & (df['prev_claim'] == df['prev_true'])
    df['Liar'] = (df['curr_claim'] != df['curr_true'])
    df['Reformed'] = (df['curr_claim'] == df['curr_true']) & (df['prev_claim'] != df['prev_true'])

    # 1. Save Plot
    save_trust_plot_pdf(df, filename="casino_hybrid_results.pdf")

    # 2. Print Summary Metrics
    print("\n" + "="*50)
    print("        RESULTS: HYBRID LOGIC")
    print("="*50)
    print(f"1. Consistent Honest: {df[df['Honest']==True]['Learned_Trust'].mean():.4f}")
    print(f"2. Current Liar:      {df[df['Liar']==True]['Learned_Trust'].mean():.4f}")
    print(f"3. Reformed Liar:     {df[df['Reformed']==True]['Learned_Trust'].mean():.4f}")
    
    # 3. Print Qualitative Examples
    print_qualitative_examples(df)

if __name__ == "__main__":
    run_hybrid_experiment()