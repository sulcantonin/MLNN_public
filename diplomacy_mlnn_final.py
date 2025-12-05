#!/usr/bin/env python
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import re
import argparse
import random
import numpy as np
import copy
import matplotlib.pyplot as plt

# --- Configuration Defaults ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
LEARNING_RATE = 0.01
EPOCHS = 400
LAMBDA_CONTRADICTION = 0.5
LAMBDA_SPARSITY = 0.05
SOFTMIN_TAU = 0.1
WEAK_GROUNDING_WEIGHT = 0.1
PLAYERS = ["ENGLAND", "GERMANY", "FRANCE", "AUSTRIA", "ITALY", "RUSSIA", "TURKEY"]
PLAYER_TO_IDX = {name: i for i, name in enumerate(PLAYERS)}
NUM_WORLDS = len(PLAYERS)

# --- 1. Data Preprocessing ---

def parse_game_log(game_file_path, holdout_ratio=0.0):
    if not os.path.exists(game_file_path):
        raise FileNotFoundError(f"Game file not found: {game_file_path}")

    with open(game_file_path, 'r') as f:
        game_data = json.load(f)

    # Define Propositions (Attacks)
    prop_map = {}
    prop_idx_counter = 0
    for i_name in PLAYERS:
        for j_name in PLAYERS:
            if i_name == j_name: continue
            prop_map[('Attacks', i_name, j_name)] = prop_idx_counter
            prop_idx_counter += 1
    
    num_propositions = len(prop_map)
    ground_truth_propositions = torch.zeros(num_propositions)

    # Regex for moves
    move_regex = re.compile(r"(?:A|F)\s+([\w/]+)\s*-\s*([\w/]+)")
    territory_owners = {}
    
    messages = []
    phases = game_data.get('phases', [])
    
    # Determine split index for holdout
    split_idx = int(len(phases) * (1 - holdout_ratio))
    
    train_messages = []
    test_messages = []

    for p_idx, phase in enumerate(phases):
        phase_name = phase.get('name')
        is_holdout = p_idx >= split_idx

        # Extract messages
        for msg in phase.get('messages', []):
            if msg.get('recipient') != 'ALL' and msg.get('sender') in PLAYERS:
                m_obj = {
                    'phase': phase_name,
                    'sender': msg.get('sender'),
                    'recipient': msg.get('recipient'),
                    'text': msg.get('message')
                }
                if is_holdout:
                    test_messages.append(m_obj)
                else:
                    train_messages.append(m_obj)
        
        # Extract Orders (Ground Truth Logic)
        orders = phase.get('orders', {})
        current_moves = {}
        
        for player, player_orders in orders.items():
            if player not in PLAYERS: continue
            for order_str in player_orders:
                match = move_regex.search(order_str)
                if match:
                    groups = match.groups()
                    if len(groups) == 2 and groups[0] and groups[1]:
                        unit_loc, target_loc = groups
                        current_moves[player] = target_loc
                        if target_loc in territory_owners:
                            victim = territory_owners[target_loc]
                            if victim != player:
                                prop = ('Attacks', player, victim)
                                if prop in prop_map:
                                    ground_truth_propositions[prop_map[prop]] = 1.0
        
        for player, target_loc in current_moves.items():
             territory_owners[target_loc] = player

    return pd.DataFrame(train_messages), pd.DataFrame(test_messages), ground_truth_propositions, prop_map

# --- 2. Model Components ---

class AccessibilityHead(nn.Module):
    def __init__(self, embedding_dim, num_worlds):
        super().__init__()
        self.num_worlds = num_worlds
        self.trust_net = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.trust_bias = nn.Parameter(torch.full((num_worlds, num_worlds), -2.0))

    def forward(self, messages_df, text_to_embedding):
        device = self.trust_bias.device 
        A_theta_logits = torch.zeros(self.num_worlds, self.num_worlds, device=device)

        # Optimization: Group messages by sender-receiver pair to batch processing
        if not text_to_embedding or messages_df.empty:
             # Enforce diagonal even if empty
            eye = torch.eye(self.num_worlds, device=device) * 10.0
            return self.trust_bias + eye

        grouped = messages_df.groupby(['sender', 'recipient'])['text'].apply(list).to_dict()

        for i in range(self.num_worlds):
            for j in range(self.num_worlds):
                if i == j:
                    A_theta_logits[i, i] = 10.0
                    continue
                
                # Messages FROM j TO i determine if i trusts j
                msgs = grouped.get((PLAYERS[j], PLAYERS[i]), [])
                
                if not msgs:
                    A_theta_logits[i, j] = self.trust_bias[i, j]
                    continue

                msg_embeddings = torch.stack([text_to_embedding[txt] for txt in msgs]).to(device)
                agg_embedding = torch.mean(msg_embeddings, dim=0)
                trust_score = self.trust_net(agg_embedding)
                A_theta_logits[i, j] = trust_score + self.trust_bias[i, j]

        return A_theta_logits

class DiplomacyMLNN(nn.Module):
    def __init__(self, embedding_dim, num_props, num_worlds, tau=0.1):
        super().__init__()
        self.accessibility_head = AccessibilityHead(embedding_dim, num_worlds)
        self.belief_propositions = nn.Parameter(torch.randn(num_props, num_worlds))
        self.tau = tau
    
    def get_belief(self, prop_idx, world_idx):
        return torch.sigmoid(self.belief_propositions[prop_idx, world_idx])

    def op_implies(self, p, q):
        p = p.clamp(1e-6, 1.0 - 1e-6)
        q = q.clamp(1e-6, 1.0 - 1e-6)
        return 1 - p + p * q

    def op_necessity(self, prop_idx, world_idx, A_theta_sigmoid):
        A_i = A_theta_sigmoid[world_idx, :] 
        B_p = torch.sigmoid(self.belief_propositions[prop_idx, :])
        implication_terms = self.op_implies(A_i, B_p)
        
        mask = torch.ones(A_i.shape[0], dtype=torch.bool, device=A_i.device)
        mask[world_idx] = False
        non_reflexive_terms = implication_terms[mask]

        if non_reflexive_terms.numel() == 0: return torch.tensor(0.5, device=A_i.device)
        
        terms = non_reflexive_terms.clamp(1e-6, 1.0)
        log_terms = terms.log()
        log_k_wp = -self.tau * torch.logsumexp(-log_terms / self.tau, dim=-1)
        return torch.exp(log_k_wp)

    def forward(self, messages_df, text_to_embedding):
        num_props, num_worlds = self.belief_propositions.shape
        A_theta_logits = self.accessibility_head(messages_df, text_to_embedding)
        A_theta_sigmoid = torch.sigmoid(A_theta_logits)
        
        total_contradiction_loss = 0
        for w_idx in range(num_worlds):
            for p_idx in range(num_props):
                belief_B_pw = self.get_belief(p_idx, w_idx)
                knowledge_K_wp = self.op_necessity(p_idx, w_idx, A_theta_sigmoid)
                total_contradiction_loss += (belief_B_pw - knowledge_K_wp)**2

        avg_contradiction_loss = total_contradiction_loss / (num_props * num_worlds)
        
        off_diag_mask = ~torch.eye(num_worlds, dtype=torch.bool, device=A_theta_logits.device)
        off_diag_probs = A_theta_sigmoid[off_diag_mask]
        sparsity_loss = torch.norm(off_diag_probs, p=1) / off_diag_mask.sum()
        
        return avg_contradiction_loss, A_theta_logits, sparsity_loss

# --- 3. Helpers ---

def train_model(model, optimizer, messages_df, text_to_embedding, ground_truth_props, prop_map, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        contra_loss, _, sparsity_loss = model(messages_df, text_to_embedding)
        
        grounding_loss = 0
        for prop, p_idx in prop_map.items():
            _, attacker, _ = prop
            attacker_idx = PLAYER_TO_IDX[attacker]
            reality_Vp = ground_truth_props[p_idx]
            
            for w_idx in range(NUM_WORLDS):
                world_belief = model.get_belief(p_idx, w_idx)
                if w_idx == attacker_idx:
                    grounding_loss += (world_belief - reality_Vp)**2
                elif reality_Vp == 1.0:
                    grounding_loss += WEAK_GROUNDING_WEIGHT * (world_belief - reality_Vp)**2
        
        avg_grounding_loss = grounding_loss / (len(prop_map) * NUM_WORLDS)
        total_loss = avg_grounding_loss + LAMBDA_CONTRADICTION * contra_loss + LAMBDA_SPARSITY * sparsity_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return contra_loss.item()

def get_embeddings(df, model, device):
    if df.empty: return {}
    unique_msgs = df['text'].unique().tolist()
    with torch.no_grad():
        embeddings = model.encode(unique_msgs, convert_to_tensor=True, device=device)
    return {txt: emb for txt, emb in zip(unique_msgs, embeddings)}

# --- 4. Modes ---

def run_permutation_test(args, device):
    print(f"\n--- Running Permutation Test (Seed: {args.seed}) ---")
    # 1. Train True Model
    train_df, _, gt, prop_map = parse_game_log(args.game_file)
    gt = gt.to(device)
    transformer = SentenceTransformer(EMBEDDING_MODEL, device=device)
    emb_map = get_embeddings(train_df, transformer, device)
    
    print("Training True Model...")
    model = DiplomacyMLNN(EMBEDDING_DIM, len(prop_map), NUM_WORLDS).to(device)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_model(model, opt, train_df, emb_map, gt, prop_map, EPOCHS)
    
    _, logits, _ = model(train_df, emb_map)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    # Calculate Mean Off-Diagonal Trust
    mask = ~np.eye(NUM_WORLDS, dtype=bool)
    true_mean_trust = probs[mask].mean()
    print(f"True Model Mean Off-Diagonal Trust: {true_mean_trust:.4f}")
    
    # 2. Train Shuffled Models
    shuffled_means = []
    print(f"Running {args.perm_runs} shuffled runs...")
    for i in range(args.perm_runs):
        # Shuffle text column relative to sender/receiver
        shuffled_df = train_df.copy()
        shuffled_text = shuffled_df['text'].tolist()
        random.shuffle(shuffled_text)
        shuffled_df['text'] = shuffled_text
        
        # Re-map embeddings for shuffled text
        shuffled_model = DiplomacyMLNN(EMBEDDING_DIM, len(prop_map), NUM_WORLDS).to(device)
        shuffled_opt = optim.Adam(shuffled_model.parameters(), lr=LEARNING_RATE)
        
        train_model(shuffled_model, shuffled_opt, shuffled_df, emb_map, gt, prop_map, int(EPOCHS/2)) 
        
        _, s_logits, _ = shuffled_model(shuffled_df, emb_map)
        s_probs = torch.sigmoid(s_logits).detach().cpu().numpy()
        shuffled_means.append(s_probs[mask].mean())
        print(f"  Run {i+1}: {shuffled_means[-1]:.4f}")
        
    print("\n--- Permutation Results for Paper ---")
    print(f"True Value: {true_mean_trust}")
    print(f"Shuffled Values: {shuffled_means}")

    # --- FIGURE GENERATION EXTENSION ---
    print("\nGenerating Figure 6 boxplot...")
    plt.figure(figsize=(8, 6))
    plt.boxplot(shuffled_means, labels=['Shuffled A_theta'], patch_artist=True, 
                boxprops=dict(facecolor='lightblue'))
    plt.scatter(1, true_mean_trust, color='red', marker='*', s=200, zorder=10, label='True A_theta')
    plt.title(f'Permutation Test: Trust Matrix Sparsity\n(Game: {os.path.basename(args.game_file)})')
    plt.ylabel('Mean Off-Diagonal Trust (0.0 - 1.0)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    output_filename = 'diplomacy_permutation_test.png'
    plt.savefig(output_filename)
    print(f"Figure saved to: {output_filename}")

def run_holdout_test(args, device):
    print(f"\n--- Running Holdout Test (Seed: {args.seed}) ---")
    train_df, test_df, gt, prop_map = parse_game_log(args.game_file, holdout_ratio=0.2)
    gt = gt.to(device)
    transformer = SentenceTransformer(EMBEDDING_MODEL, device=device)
    
    # Compute embeddings for ALL messages (train + test)
    all_msgs = pd.concat([train_df, test_df])
    emb_map = get_embeddings(all_msgs, transformer, device)
    
    # 1. Train on Train Set
    print(f"Training on {len(train_df)} messages...")
    model = DiplomacyMLNN(EMBEDDING_DIM, len(prop_map), NUM_WORLDS).to(device)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_model(model, opt, train_df, emb_map, gt, prop_map, EPOCHS)
    
    # 2. Evaluate L_contra on Test Set using Learned A_theta
    model.eval()
    learned_loss, _, _ = model(test_df, emb_map)
    print(f"Learned A_theta Holdout Loss: {learned_loss.item():.6f}")
    
    # 3. Evaluate Identity Matrix (Baseline)
    with torch.no_grad():
        identity_model = copy.deepcopy(model)
        identity_model.accessibility_head.trust_bias.data.fill_(-10.0)
        identity_model.accessibility_head.trust_bias.data.fill_diagonal_(10.0)
        for param in identity_model.accessibility_head.trust_net.parameters():
            param.fill_(0.0)
        
        identity_loss, _, _ = identity_model(test_df, emb_map)
        print(f"Identity Matrix Holdout Loss: {identity_loss.item():.6f}")

    # 4. Evaluate Shuffled Matrix (Baseline)
    shuffled_df = train_df.copy()
    shuffled_text = shuffled_df['text'].tolist()
    random.shuffle(shuffled_text)
    shuffled_df['text'] = shuffled_text
    
    shuff_model = DiplomacyMLNN(EMBEDDING_DIM, len(prop_map), NUM_WORLDS).to(device)
    shuff_opt = optim.Adam(shuff_model.parameters(), lr=LEARNING_RATE)
    train_model(shuff_model, shuff_opt, shuffled_df, emb_map, gt, prop_map, EPOCHS)
    
    shuff_model.eval()
    shuffled_loss, _, _ = shuff_model(test_df, emb_map)
    print(f"Shuffled A_theta Holdout Loss: {shuffled_loss.item():.6f}")

def run_normal(args, device):
    train_df, _, gt, prop_map = parse_game_log(args.game_file)
    gt = gt.to(device)
    transformer = SentenceTransformer(EMBEDDING_MODEL, device=device)
    emb_map = get_embeddings(train_df, transformer, device)
    
    model = DiplomacyMLNN(EMBEDDING_DIM, len(prop_map), NUM_WORLDS).to(device)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Training on {len(train_df)} messages from {args.game_file}...")
    train_model(model, opt, train_df, emb_map, gt, prop_map, EPOCHS)
    
    _, logits, _ = model(train_df, emb_map)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    
    print("\n--- Final Trust Matrix (Copy to Table) ---")
    print("      " + " ".join([f"{p[:3]:>5}" for p in PLAYERS]))
    for i, p in enumerate(PLAYERS):
        row = [f"{probs[i,j]:.2f}" for j in range(NUM_WORLDS)]
        print(f"{p[:3]:<5} " + " ".join([f"{r:>5}" for r in row]))

# --- Main ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_file", type=str, required=True, help="Path to .json game file")
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "permutation", "holdout"], help="Experiment mode")
    parser.add_argument("--perm_runs", type=int, default=20, help="Number of permutation runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Robust Seeding for Reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Force deterministic algorithms on GPU (slower but exact)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} | Mode: {args.mode} | Game: {args.game_file} | Seed: {args.seed}")

    if args.mode == "permutation":
        run_permutation_test(args, device)
    elif args.mode == "holdout":
        run_holdout_test(args, device)
    else:
        run_normal(args, device)