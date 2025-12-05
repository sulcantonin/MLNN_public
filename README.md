Here is a comprehensive `README.md` designed for your repository. It organizes the files you provided into logical sections corresponding to the experiments in the `MLNN.pdf` paper, provides installation instructions, and explains how to run each component.

-----

# Modal Logical Neural Networks (MLNN)

This repository contains the official implementation and experiments for the paper **"Modal Logical Neural Networks"**.

**MLNN** is a neurosymbolic framework that integrates deep learning with the formal semantics of modal logic. By leveraging differentiable Kripke semantics, MLNNs allow neural networks to reason about **necessity** ($\square$) and **possibility** ($\diamond$) across multiple "possible worlds."

The framework supports two modes of operation:

1.  **Deductive (Fixed Structure):** Enforcing user-defined logical guardrails (e.g., grammatical rules, abstention logic).
2.  **Inductive (Learned Structure):** Learning the accessibility relation ($A_\theta$) between worlds from data (e.g., learning trust networks in multi-agent games).

-----

## 📂 Repository Structure

The codebase is organized by the experiments presented in the paper:

| File | Type | Description | Paper Section |
| :--- | :--- | :--- | :--- |
| `MLNN_POS_PAPER.ipynb` | Notebook | **Grammatical Guardrailing:** Enforcing strict POS tagging rules over a BiLSTM using logical contradiction loss. | Sec 5.1 |
| `MLNN_DIALECTS_MLNN_CP.ipynb` | Notebook | **Logical Indeterminacy:** Using axioms to force abstention on "Neutral" dialect inputs. Compares MLNN vs. Conformal Prediction. | Sec 5.2 |
| `MLNN_TEMPORAL_EPISTEMIC.ipynb` | Notebook | **Toy Epistemic Learning:** A controlled environment demonstrating the learning of $A_\theta$ to resolve a logical contradiction in a temporal-epistemic setting. | Sec 5.3 |
| `diplomacy_mlnn_final.py` | Script | **Diplomacy Trust Modeling:** Learning latent trust structures in Diplomacy game logs by checking consistency between messages and actions. | Sec 5.4 |
| `casino-mlnn.py` | Script | **Negotiation Deception:** Detecting deception in the CaSiNo dataset by enforcing temporal consistency axioms on agent history. | Sec 5.5 |
| `ablation_final.py` | Script | **Synthetic Ablation:** Recovering a ground-truth "Ring" topology to validate the inductive learning capabilities of $A_\theta$. | Sec 5.6 |
| `MLNN.pdf` | Doc | The full research paper. | - |

-----

## 🛠️ Installation

The code requires Python 3.8+ and PyTorch.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/MLNN.git
    cd MLNN
    ```

2.  **Install dependencies:**

    ```bash
    pip install torch numpy pandas matplotlib scikit-learn tqdm nltk sentence-transformers datasets
    ```

-----

## Running the Experiments

### 1. Grammatical Guardrailing (POS Tagging)

This experiment demonstrates using MLNNs to enforce 10 rigid grammatical axioms (e.g., "A determiner cannot be followed by a verb") on a sequence tagger.

  * **Run:** Open `MLNN_POS_PAPER.ipynb` in Jupyter/Colab.
  * **Output:** Generates violation statistics and accuracy trade-off plots comparing a Baseline BiLSTM vs. MLNN ($\beta=1.0$).

### 2. Logical Indeterminacy (Dialect Classification)

This experiment compares MLNNs against Conformal Prediction (CP) in handling "Neutral" inputs that were never seen during training.

  * **Run:** Open `MLNN_DIALECTS_MLNN_CP.ipynb`.
  * **Key Result:** The MLNN achieves near-perfect selective accuracy on the unknown class via axiomatic definition, whereas CP trades off coverage and risk.

### 3. Epistemic Trust in Diplomacy

This script processes Diplomacy game logs to learn which agents trust each other based on whether their actions align with their messages.

  * **Data:** Requires a Diplomacy game JSON file (compatible with the format used in Bakhtin et al., 2022).
  * **Run:**
    ```bash
    python diplomacy_mlnn_final.py --game_file path/to/game.json --mode normal
    ```
  * **Modes:**
      * `normal`: Trains and prints the learned trust matrix.
      * `permutation`: Runs permutation tests to validate structural significance (Figure 6 in paper).
      * `holdout`: Tests generalization on unseen game phases.

### 4. Deception in Negotiation (CaSiNo)

This script uses the HuggingFace `kchawla123/casino` dataset to detect inconsistent/deceptive negotiation strategies.

  * **Run:**
    ```bash
    python casino-mlnn.py
    ```
  * **Output:** Generates `casino_hybrid_results.pdf` showing trust distributions for Honest vs. Deceptive agents.

### 5. Synthetic Structure Learning (Ablation)

Validates that $A_\theta$ can recover a ground-truth "Ring" structure purely from logical constraints.

  * **Run:**
    ```bash
    python ablation_final.py
    ```
  * **Output:** Prints MSE (Mean Squared Error) of the recovered structure against ground truth for various $k$ (Top-K) and $\tau$ (Temperature) settings.

-----

## Key Concepts

### The Learnable Accessibility Relation ($A_\theta$)

Unlike standard modal logic where the accessibility relation $R$ is fixed, MLNNs parameterize it as a neural network.

  * **Input:** State embeddings (e.g., sentence embeddings, agent histories).
  * **Output:** A soft adjacency matrix $A_\theta \in [0,1]^{W \times W}$.
  * **Training:** Optimized end-to-end by minimizing the **Logical Contradiction Loss**.

### Differentiable Modal Operators

We implement $\square \phi$ (Necessity) and $\diamond \phi$ (Possibility) using smooth approximations of `min` and `max` to allow gradient flow:

  * $\square \phi \approx \text{SoftMin}((1 - A_{ij}) + \text{Truth}(\phi_j))$
  * $\diamond \phi \approx \text{SoftMax}(A_{ij} \times \text{Truth}(\phi_j))$

-----

## 📜 Citation

If you use this code or framework in your research, please cite:

```bibtex
@misc{sulc2025modallogicalneuralnetworks,
      title={Modal Logical Neural Networks}, 
      author={Antonin Sulc},
      year={2025},
      eprint={2512.03491},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.03491}, 
}
```
