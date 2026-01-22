import torch.nn.functional as F
from tqdm import tqdm
from model import Encoder_overall
from preprocess import *
from preprocess import construct_hypergraph
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import random

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load graph data
graph_data = torch.load("graph_data.pt")
transcript_data = graph_data["transcript_data"]
metabolite_data = graph_data["metabolite_data"]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare features
features_transcript = transcript_data.x.float().to(device)
features_metabolite = metabolite_data.x.float().to(device)

# Prepare adjacency matrices
adj = adjacent_matrix_preprocessing(transcript_data, metabolite_data)
adj_spatial_transcript = adj['adj_spatial_omics1'].to(device)
adj_spatial_metabolite = adj['adj_spatial_omics2'].to(device)
adj_feature_transcript = adj['adj_feature_omics1'].to(device)
adj_feature_metabolite = adj['adj_feature_omics2'].to(device)

# Load spatial coordinates
df = pd.read_csv('aligned_ST_data.csv')
coordinates = torch.tensor(df.iloc[:, [1, 2]].values, dtype=torch.float32).to(device)
# Min-Max
coordinates_min = coordinates.min(dim=0, keepdim=True)[0]
coordinates_max = coordinates.max(dim=0, keepdim=True)[0]
coordinates = (coordinates - coordinates_min) / (coordinates_max - coordinates_min)
# Coordinates are aligned, share for both omics
coordinates_transcript = coordinates
coordinates_metabolite = coordinates

# Initialize the model
dim_input1 = features_transcript.shape[1]
dim_input2 = features_metabolite.shape[1]
latent_dim = 256

model = Encoder_overall(
    dim_in_feat_omics1=dim_input1,
    dim_latent_feat_omics1=latent_dim,
    dim_in_feat_omics2=dim_input2,
    dim_latent_feat_omics2=latent_dim,
    dim_output_recon1=dim_input1,
    dim_output_recon2=dim_input2
).to(device)

# Optimizer
learning_rate = 0.0001
weight_decay = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Hyperparameters
epochs = 300
weight_factors = [2, 2, 1, 1,1]  # Adjust weights for loss components

#Construct hypergraph
H = construct_hypergraph(features_transcript, features_metabolite, n_neighbors=10)

# Contrastive Loss (InfoNCE)
def contrastive_loss(emb1, emb2, temperature=0.07):
    emb1 = F.normalize(emb1, dim=1)
    emb2 = F.normalize(emb2, dim=1)
    batch_size = emb1.size(0)

    sim_matrix = torch.matmul(emb1, emb2.T) / temperature
    labels = torch.arange(batch_size).to(emb1.device)

    loss_1 = F.cross_entropy(sim_matrix, labels)
    loss_2 = F.cross_entropy(sim_matrix.T, labels)

    loss = (loss_1 + loss_2) / 2
    return loss

# Training loop
for epoch in tqdm(range(epochs)):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    results =  model(
        features_omics1=features_transcript,
        features_omics2=features_metabolite,
        coordinates_omics1=coordinates,
        coordinates_omics2=coordinates,
        adj_spatial_omics1=adj_spatial_transcript,
        adj_feature_omics1=adj_feature_transcript,
        adj_spatial_omics2=adj_spatial_metabolite,
        adj_feature_omics2=adj_feature_metabolite,
        H=H
    )

    # Reconstruction loss
    loss_recon_transcript = F.mse_loss(features_transcript, results['emb_recon_omics1'])
    loss_recon_metabolite = F.mse_loss(features_metabolite, results['emb_recon_omics2'])

    # Ensure dimensions match for latent embeddings
    min_size_transcript = min(results['emb_latent_omics1'].size(0), results['emb_latent_omics1_across_recon'].size(0))
    results['emb_latent_omics1'] = results['emb_latent_omics1'][:min_size_transcript, :]
    results['emb_latent_omics1_across_recon'] = results['emb_latent_omics1_across_recon'][:min_size_transcript, :]

    min_size_metabolite = min(results['emb_latent_omics2'].size(0), results['emb_latent_omics2_across_recon'].size(0))
    results['emb_latent_omics2'] = results['emb_latent_omics2'][:min_size_metabolite, :]
    results['emb_latent_omics2_across_recon'] = results['emb_latent_omics2_across_recon'][:min_size_metabolite, :]

    # Correspondence loss
    loss_corr_transcript = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_omics1_across_recon'])
    loss_corr_metabolite = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_omics2_across_recon'])

    #contrastive loss
    loss_contrastive = contrastive_loss(results['emb_latent_omics1'], results['emb_latent_omics2'])
    weight_contrastive = 1

    # Dynamic weight adjustment
    if epoch > 0:
        recon_ratio = loss_recon_transcript.item() / (loss_recon_metabolite.item() + 1e-8)
        recon_ratio = min(max(recon_ratio, 0.5), 2)
        weight_factors[1] = weight_factors[0] * recon_ratio

        corr_ratio = loss_corr_transcript.item() / (loss_corr_metabolite.item() + 1e-8)
        corr_ratio = min(max(corr_ratio, 0.5), 2)
        weight_factors[3] = weight_factors[2] * corr_ratio

        contrastive_ratio = (loss_corr_transcript.item() + loss_corr_metabolite.item()) / (
                    loss_contrastive.item() + 1e-8)
        contrastive_ratio = min(max(contrastive_ratio, 0.5), 2)
        weight_contrastive = 0.1 * contrastive_ratio

    loss = (
        weight_factors[0] * loss_recon_transcript +
        weight_factors[1] * loss_recon_metabolite +
        weight_factors[2] * loss_corr_transcript +
        weight_factors[3] * loss_corr_metabolite+
        weight_contrastive * loss_contrastive
    )

    # Backward and optimize
    loss.backward()
    optimizer.step()

    # Log training losses
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print(f"  Loss recon transcript: {loss_recon_transcript.item():.4f}")
    print(f"  Loss recon metabolite: {loss_recon_metabolite.item():.4f}")
    print(f"  Loss corr transcript: {loss_corr_transcript.item():.4f}")
    print(f"  Loss corr metabolite: {loss_corr_metabolite.item():.4f}")
    print(f"  loss_contrastive: {loss_contrastive.item():.4f}")
    print(f"  Total Loss: {loss.item():.4f}")

    # Save training log
    loss_log = {
        "epoch": epoch + 1,
        "loss_recon_transcript": loss_recon_transcript.item(),
        "loss_recon_metabolite": loss_recon_metabolite.item(),
        "loss_corr_transcript": loss_corr_transcript.item(),
        "loss_corr_metabolite": loss_corr_metabolite.item(),
        "loss_contrastive": loss_contrastive.item(),
        "total_loss": loss.item(),
        "weight_factors": weight_factors,
    }

    with open("training_log.txt", "a") as f:
        f.write(f"{loss_log}\n")

    # Evaluate the model every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
        model.eval()
        with torch.no_grad():
            eval_results = model(
        features_omics1=features_transcript,
        features_omics2=features_metabolite,
        coordinates_omics1=coordinates,
        coordinates_omics2=coordinates,
        adj_spatial_omics1=adj_spatial_transcript,
        adj_feature_omics1=adj_feature_transcript,
        adj_spatial_omics2=adj_spatial_metabolite,
        adj_feature_omics2=adj_feature_metabolite,
        H=H
    )

            # Single-modality reconstruction MAE
            mae_transcript = mean_absolute_error(
                features_transcript.cpu(), eval_results['emb_recon_omics1'].cpu())
            mae_metabolite = mean_absolute_error(
                features_metabolite.cpu(), eval_results['emb_recon_omics2'].cpu())

            # Cross-modality reconstruction MAE
            mae_transcript_to_metabolite = mean_absolute_error(
                eval_results['emb_latent_omics1'].cpu(),
                eval_results['emb_latent_omics1_across_recon'].cpu())
            mae_metabolite_to_transcript = mean_absolute_error(
                eval_results['emb_latent_omics2'].cpu(),
                eval_results['emb_latent_omics2_across_recon'].cpu())

            # Print evaluation metrics
            print(f"\n===== Evaluation at Epoch {epoch + 1} =====")
            print(f"  MAE Transcriptomics (self): {mae_transcript:.4f}")
            print(f"  MAE Metabolomics (self): {mae_metabolite:.4f}")
            print(f"  MAE Transcript → Metabolite: {mae_transcript_to_metabolite:.4f}")
            print(f"  MAE Metabolite → Transcript: {mae_metabolite_to_transcript:.4f}")
            print(f"==========================================\n")

            # Optionally save evaluation results per 10 epochs
            with open("evaluation_metrics_per_10epochs.txt", "a") as f:
                f.write(
                    f"Epoch {epoch+1},"
                    f"{mae_transcript:.4f},"
                    f"{mae_metabolite:.4f},"
                    f"{mae_transcript_to_metabolite:.4f},"
                    f"{mae_metabolite_to_transcript:.4f}\n"
                )

        model.train()  # Switch back to training mode

print("Training and periodic evaluation complete.")
# Evaluate the model
model.eval()
with torch.no_grad():
    results = model(
        features_omics1=features_transcript,
        features_omics2=features_metabolite,
        coordinates_omics1=coordinates,
        coordinates_omics2=coordinates,
        adj_spatial_omics1=adj_spatial_transcript,
        adj_feature_omics1=adj_feature_transcript,
        adj_spatial_omics2=adj_spatial_metabolite,
        adj_feature_omics2=adj_feature_metabolite,
        H=H
    )

    emb_latent_transcript = F.normalize(results['emb_latent_omics1'], p=2, dim=1).cpu().numpy()
    emb_latent_metabolite = F.normalize(results['emb_latent_omics2'], p=2, dim=1).cpu().numpy()
    emb_combined = F.normalize(results['emb_latent_combined'], p=2, dim=1).cpu().numpy()

    emb_recon_transcript = results['emb_recon_omics1'].cpu().numpy()
    emb_recon_metabolite = results['emb_recon_omics2'].cpu().numpy()

    emb_transcript_from_metabolite = results['emb_latent_omics1_across_recon'].cpu().numpy()
    emb_metabolite_from_transcript = results['emb_latent_omics2_across_recon'].cpu().numpy()

torch.save({
    # Normalized latent embeddings
    "emb_latent_transcript": emb_latent_transcript,
    "emb_latent_metabolite": emb_latent_metabolite,
    "emb_combined": emb_combined,

    # Reconstructed original features
    "emb_recon_transcript": emb_recon_transcript,
    "emb_recon_metabolite": emb_recon_metabolite,

    # Cross-modal inferred reconstructions
    "emb_transcript_from_metabolite": emb_transcript_from_metabolite,
    "emb_metabolite_from_transcript": emb_metabolite_from_transcript,

}, "SHINE embeddings.pt")

print("Saved successfully.")
