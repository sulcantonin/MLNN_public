import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- 1. The Model (MLNN with Top-K and Learnable Toggle) ---
class StructuralMLNN(nn.Module):
    def __init__(self, observations, num_agents, tau=0.1, top_k=None, learnable_structure=True):
        super().__init__()
        self.num_agents = num_agents
        self.tau = tau
        self.top_k = top_k
        
        # Initialization: Random logits
        # If learnable=False, this represents a fixed random logic (which should fail the task)
        self.logits = nn.Parameter(torch.randn(num_agents, num_agents))
        
        if not learnable_structure:
            self.logits.requires_grad = False

        # Fixed Perception (Observations)
        self.register_buffer('beliefs', observations)

    def get_access(self):
        # 1. Sigmoid for [0,1] bounds
        # We add eye to encourage self-loops initially, though strictly optional
        probs = torch.sigmoid(self.logits)
        
        # 2. Apply Top-K Mask (Section 3.3 of paper)
        if self.top_k is not None and self.top_k < self.num_agents:
            # Find the k-th largest value in each row
            top_vals, _ = torch.topk(probs, self.top_k, dim=1)
            kth_val = top_vals[:, -1].unsqueeze(1)
            
            # Create a hard mask (1 if >= kth_val, else 0)
            mask = (probs >= kth_val).float()
            
            # Apply mask. 
            # Note: In a full production implementation, we might use Straight-Through Estimator
            # for the mask gradients, but for this demo, masking the values works for forward pass.
            probs = probs * mask
            
            # Re-normalize to ensure row sum doesn't vanish? 
            # The paper implies using weights directly, so we leave as is.

        return probs

    def forward(self, beacon_start_idx):
        A = self.get_access()
        B = self.beliefs
        
        # 1. Consistency Loss (Box): "Don't trust those who contradict facts"
        general_facts = B[:beacon_start_idx, :]
        disagreement = torch.cdist(general_facts.T, general_facts.T, p=1) / beacon_start_idx
        loss_box = torch.mean(A * disagreement)

        # 2. Expansion Loss (Diamond): "Find someone who has the beacon"
        loss_diamond = torch.tensor(0.0)
        
        # Vectorized implementation of the diamond loss loop for speed
        # We need to check: For every agent i, does there exist a neighbor j such that B[beacon_i, j] is true?
        beacon_indices = torch.arange(self.num_agents) + beacon_start_idx
        
        # Extract the specific beacon rows needed by each agent
        # Shape: (num_agents, num_agents) -> (Target Beacon for Agent i, potential neighbor j)
        target_facts = B[beacon_indices, :] 
        
        # Weighted Evidence: A[i,j] * Fact[j]
        weighted_evidence = A * target_facts
        
        # Softmax (differentiable max)
        # logsumexp trick: tau * log(sum(exp(x/tau)))
        max_evidence = self.tau * torch.logsumexp(weighted_evidence / self.tau, dim=1)
        
        # We want max_evidence to be 1.0
        loss_diamond = torch.mean((1.0 - max_evidence)**2)

        return loss_box + loss_diamond, A

# --- 2. Data Generation ---
def generate_ring_data(num_agents=20, num_props=100):
    """
    Generates a larger ring for N=20 to make Top-K meaningful.
    """
    # 1. Smooth "Ring of Truth"
    V_gt = torch.rand(num_props, num_agents)
    for _ in range(5): 
        for i in range(num_agents):
            neighbor = (i + 1) % num_agents
            V_gt[:, i] = 0.8 * V_gt[:, i] + 0.2 * V_gt[:, neighbor]
    
    V_gt = V_gt.round()

    # 2. Add Beacons
    beacon_start_idx = num_props
    beacons = torch.zeros(num_agents, num_agents)
    for i in range(num_agents):
        target = (i + 1) % num_agents
        beacons[i, target] = 1.0 
        
    V_final = torch.cat([V_gt, beacons], dim=0)
    return V_final, beacon_start_idx

# --- 3. Ablation Suite ---
def train_and_evaluate(config, seeds=3):
    losses = []
    mses = []
    
    num_agents = 20
    
    # Ground Truth Structure (Ring) for metric calculation
    A_gt = torch.eye(num_agents)
    for i in range(num_agents): A_gt[i, (i+1)%num_agents] = 1.0
    mask_off_diag = ~torch.eye(num_agents, dtype=torch.bool)

    for seed in range(seeds):
        torch.manual_seed(seed)
        V_final, beacon_idx = generate_ring_data(num_agents)
        
        model = StructuralMLNN(
            V_final, 
            num_agents, 
            tau=config['tau'], 
            top_k=config['top_k'], 
            learnable_structure=config['learnable']
        )
        
        # If not learnable, optimizer won't find params, that's fine (0 step)
        params = [p for p in model.parameters() if p.requires_grad]
        if len(params) > 0:
            optimizer = optim.Adam(params, lr=0.05)
            
            for epoch in range(200):
                optimizer.zero_grad()
                loss, _ = model(beacon_idx)
                loss.backward()
                optimizer.step()
        else:
            # Just run forward once to get loss
            pass

        # Final Eval
        loss, A_pred = model(beacon_idx)
        A_final = A_pred.detach()
        
        # Metric: MSE against Ground Truth Ring (on off-diagonals)
        mse = torch.mean((A_final[mask_off_diag] - A_gt[mask_off_diag])**2).item()
        
        losses.append(loss.item())
        mses.append(mse)

        # Add this block after the training loop in ablation_final.py

    # ... [Training Loop] ...
        
    # 1. Extract Final Learned Matrix
    _, A_pred = model(beacon_idx)
    A_final = A_pred.detach().cpu().numpy()
        
    # 2. Define Ground Truth (Ring Structure for N=5)
    # GT: Agent i trusts Agent i (Reflexive) and Agent i+1 (Functional)
    A_gt = np.eye(num_agents)
    for i in range(num_agents):
        A_gt[i, (i+1)%num_agents] = 1.0
        
    # 3. Print Comparative Analysis for Agent 0
    print(f"\n{'Relation':<20} | {'Ground Truth':<12} | {'Estimated':<12}")
    print("-" * 50)
    for j in range(num_agents):
        relation_str = f"Trust(0 -> {j})"
        print(f"{relation_str:<20} | {A_gt[0,j]:.2f}         | {A_final[0,j]:.2f}")

    return np.mean(mses), np.std(mses), np.mean(losses), np.std(losses)

def run_ablation_suite():
    print(f"{'Setting':<20} | {'Param':<15} | {'MSE (Mean ± Std)':<25} | {'Loss (Mean ± Std)':<25}")
    print("-" * 90)

    # 1. Temperature Sweep (Fix k=8, Learnable=True)
    # Paper Section 5.5: tau in {0.05, 0.1, 0.2}
    taus = [0.05, 0.1, 0.2]
    for t in taus:
        mse_m, mse_s, loss_m, loss_s = train_and_evaluate({'tau': t, 'top_k': 8, 'learnable': True})
        print(f"{'Temperature':<20} | {t:<15} | {mse_m:.4f} ± {mse_s:.4f}          | {loss_m:.4f} ± {loss_s:.4f}")

    print("-" * 90)

    # 2. Top-k Mask Sweep (Fix tau=0.1, Learnable=True)
    # Paper Section 5.5: k in {4, 8, 16}
    # Note: N=20, so k=16 is dense, k=4 is sparse
    ks = [4, 8, 16]
    for k in ks:
        mse_m, mse_s, loss_m, loss_s = train_and_evaluate({'tau': 0.1, 'top_k': k, 'learnable': True})
        print(f"{'Top-k Mask':<20} | {k:<15} | {mse_m:.4f} ± {mse_s:.4f}          | {loss_m:.4f} ± {loss_s:.4f}")

    print("-" * 90)

    # 3. Learnable vs Fixed (Fix tau=0.1, k=8)
    # Fixed R vs Learnable A_theta
    settings = [
        {'name': 'Fixed R', 'learnable': False},
        {'name': 'Learnable A', 'learnable': True}
    ]
    for s in settings:
        mse_m, mse_s, loss_m, loss_s = train_and_evaluate({'tau': 0.1, 'top_k': 8, 'learnable': s['learnable']})
        print(f"{'Relation':<20} | {s['name']:<15} | {mse_m:.4f} ± {mse_s:.4f}          | {loss_m:.4f} ± {loss_s:.4f}")

if __name__ == "__main__":
    run_ablation_suite()