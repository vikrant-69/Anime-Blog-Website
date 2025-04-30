import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# --- Configuration (Based on Paper's Findings/Defaults) ---
K_NEIGHBORS = 4  # Number of neighbors for LLR 
EMBEDDING_DIM = 16 # Dimension Fg of the embedding space 
LAMBDA_REG = 1e-5 # Regularization for LLR (Paper doesn't specify, using a small value)
MU_SELECT = 0.15   # Selection interval threshold 
LAMBDA_EPC = 1.0   # Weight for EPC loss 
LAMBDA_GAZE = 1.0  # Weight for Gaze loss 

# --- Model Definition (DAGEN Structure) ---

class FeatureExtractor(nn.Module):
    """ Feature extractor based on ResNet-18 followed by MLP """
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        # Using a pre-trained ResNet-18 as backbone 
        resnet = models.resnet18(pretrained=True)
        # Remove the final classification layer
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        # MLP layer 
        self.mlp = nn.Linear(resnet.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        embedding = self.mlp(x)
        return embedding

class GazePredictor(nn.Module):
    """ Simple linear mapping from embedding to gaze direction """
    def __init__(self, embedding_dim=EMBEDDING_DIM, gaze_dim=2): # gaze_dim=2 for (yaw, pitch) 
        super().__init__()
        self.fc = nn.Linear(embedding_dim, gaze_dim) # Linear mapping h 

    def forward(self, embedding):
        gaze = self.fc(embedding)
        # Normalize gaze vector (optional, but common for direction)
        # gaze = F.normalize(gaze, p=2, dim=1) # L2 normalization if needed
        return gaze

class DAGEN(nn.Module):
    """ Complete DAGEN Network """
    def __init__(self, embedding_dim=EMBEDDING_DIM, gaze_dim=2):
        super().__init__()
        self.feature_extractor = FeatureExtractor(embedding_dim)
        self.gaze_predictor = GazePredictor(embedding_dim, gaze_dim)

    def forward(self, x):
        embedding = self.feature_extractor(x)
        gaze = self.gaze_predictor(embedding)
        return embedding, gaze

# --- Loss Function Components ---

def calculate_llr_weights(target_gaze_pred, source_gaze_gt, source_indices, k, lambda_reg):
    """
    Calculates Locally Linear Representation (LLR) weights based on Eq. 5.

    Args:
        target_gaze_pred (Tensor): Predicted gaze for a single target sample (shape: [gaze_dim]).
        source_gaze_gt (Tensor): Ground truth gaze for k source neighbors (shape: [k, gaze_dim]).
        source_indices (Tensor): Indices of the k source neighbors.
        k (int): Number of neighbors.
        lambda_reg (float): L2 regularization parameter.

    Returns:
        Tensor: Optimized LLR weights W* (shape: [k]).
                Returns None if calculation fails (e.g., singular matrix).
    """
    gaze_dim = target_gaze_pred.shape[0]
    target_gaze_pred_k = target_gaze_pred.unsqueeze(0).repeat(k, 1) # Shape [k, gaze_dim]

    # Calculate local covariance matrix Sj (Eq. 4) 
    diff = target_gaze_pred_k - source_gaze_gt # Shape [k, gaze_dim]
    Sj = torch.matmul(diff.T, diff) # Shape [gaze_dim, gaze_dim]

    # Add regularization and compute inverse (Eq. 5) 
    identity = torch.eye(gaze_dim, device=Sj.device) * lambda_reg
    try:
        # Using solve is generally more stable than direct inverse
        # We want to solve (Sj + lambda*I) * x = 1_k, but need to handle the denominator
        # Let M = (Sj + lambda*I)
        M_inv = torch.linalg.inv(Sj + identity)
    except torch.linalg.LinAlgError:
        print("Warning: Singular matrix encountered in LLR weight calculation.")
        return None # Indicate failure

    ones_k = torch.ones(k, 1, device=Sj.device)

    # Calculate numerator: M_inv * 1_k (adjusting for matrix shapes)
    # We actually need the weights W, so we compute M_inv * ones_k (as vector) first
    # And use the formula from Eq. 5 directly seems more complex than needed for weights
    # Let's re-derive slightly or use the structure:
    # W_j* = normalize( (Sj + lambda*I)^-1 * 1_k ) such that sum(W_j*) = 1
    # Numerator = (S_j + lambda I)^(-1) * 1_k -> This is solving (S_j + lambda I) * W = 1_k ?? No.

    # Revisit Eq. 5 structure: num = (Sj + lambda*I)^-1 * 1_k, den = 1_k^T * num
    # The shapes are tricky. Let's use the definition from paper slightly differently:
    # Find W that minimizes || target - W @ source ||^2 + lambda * ||W||^2, s.t. sum(W)=1
    # This is a constrained least squares problem. Eq 5 gives the analytical solution.

    # Applying Eq 5 directly:
    num = torch.matmul(M_inv, ones_k) # Shape [gaze_dim, 1] ?? No, this needs careful shape analysis based on paper derivation

    # Let's rethink Sj calculation based on Eq 4 and 5 shapes.
    # G_i^s = [g_1^s, ..., g_k^s] -> Shape [gaze_dim, k] if column vectors
    # G_j^t = [g_j^t, ..., g_j^t] -> Shape [gaze_dim, k]
    # S_j = (G_j^t - G_i^s)^T @ (G_j^t - G_i^s) -> Shape [k, k]
    # This seems more likely given Wj is [w_j1, ..., w_jk] (k elements)

    G_i_s = source_gaze_gt.T # Shape [gaze_dim, k]
    G_j_t = target_gaze_pred.unsqueeze(1).repeat(1, k) # Shape [gaze_dim, k]
    S_j = torch.matmul((G_j_t - G_i_s).T, (G_j_t - G_i_s)) # Shape [k, k] 

    identity_k = torch.eye(k, device=S_j.device) * lambda_reg
    try:
        M_inv_k = torch.linalg.inv(S_j + identity_k)
    except torch.linalg.LinAlgError:
         print("Warning: Singular matrix encountered in LLR weight calculation (k x k).")
         return None # Indicate failure

    ones_k_vec = torch.ones(k, 1, device=S_j.device) # Shape [k, 1]

    # Eq. 5: W_j* = (M_inv_k @ 1_k) / (1_k^T @ M_inv_k @ 1_k) 
    numerator = torch.matmul(M_inv_k, ones_k_vec) # Shape [k, 1]
    denominator = torch.matmul(ones_k_vec.T, numerator) # Shape [1, 1]

    if torch.abs(denominator) < 1e-9: # Avoid division by zero
         print("Warning: Small denominator encountered in LLR weight calculation.")
         return None

    W_star = numerator / denominator # Shape [k, 1]
    return W_star.squeeze(1) # Return shape [k]


def epc_loss(target_embeddings, target_gaze_preds, source_embeddings, source_gaze_gt, k, lambda_reg, mu_select):
    """
    Calculates the Embedding with Prediction Consistency (EPC) loss (Eq. 9).

    Args:
        target_embeddings (Tensor): Embeddings for target batch (shape: [Bt, embed_dim]).
        target_gaze_preds (Tensor): Predictions for target batch (shape: [Bt, gaze_dim]).
        source_embeddings (Tensor): Embeddings for source batch (shape: [Bs, embed_dim]).
        source_gaze_gt (Tensor): Ground truth gaze for source batch (shape: [Bs, gaze_dim]).
        k (int): Number of neighbors.
        lambda_reg (float): L2 regularization for LLR.
        mu_select (float): Selection interval threshold.

    Returns:
        Tensor: Scalar EPC loss value.
                Returns 0 if no valid target samples are found.
    """
    Bt, embed_dim = target_embeddings.shape
    Bs, gaze_dim = source_gaze_gt.shape
    total_loss = 0.0
    valid_samples = 0

    # Calculate pairwise distances between target predictions and source ground truth in gaze space
    # Using angular distance (cosine similarity) might be more appropriate than Euclidean for gaze vectors
    # Paper uses Eq 1: max(|yaw_diff|, |pitch_diff|) < mu 
    # Let's use Euclidean distance for simplicity, adapt if needed. Assumes gaze vectors are comparable.
    # gaze_dist = torch.cdist(target_gaze_preds, source_gaze_gt, p=2) # Shape [Bt, Bs]

    # Implementing Eq 1 check
    yaw_diff = torch.abs(target_gaze_preds[:, 0].unsqueeze(1) - source_gaze_gt[:, 0].unsqueeze(0)) # Shape [Bt, Bs]
    pitch_diff = torch.abs(target_gaze_preds[:, 1].unsqueeze(1) - source_gaze_gt[:, 1].unsqueeze(0)) # Shape [Bt, Bs]
    max_diff = torch.maximum(yaw_diff, pitch_diff) # Shape [Bt, Bs]
    neighbor_mask = max_diff < mu_select # Boolean mask [Bt, Bs]

    for j in range(Bt): # Iterate through each target sample
        # Find indices of neighbors in the source batch based on Eq 1 
        neighbor_indices = torch.where(neighbor_mask[j])[0]

        if len(neighbor_indices) >= k: # Check if enough neighbors found 
            # Randomly select k neighbors if more are available 
            perm = torch.randperm(len(neighbor_indices), device=target_embeddings.device)
            selected_indices = neighbor_indices[perm[:k]]

            # Get data for selected neighbors
            k_source_gaze_gt = source_gaze_gt[selected_indices] # Shape [k, gaze_dim]
            k_source_embeddings = source_embeddings[selected_indices] # Shape [k, embed_dim]

            # Calculate LLR weights W* for this target sample 
            W_star = calculate_llr_weights(target_gaze_preds[j], k_source_gaze_gt, selected_indices, k, lambda_reg)

            if W_star is not None:
                # Calculate target hypothesis embedding (Eq. 7) 
                # W_star shape [k], k_source_embeddings shape [k, embed_dim]
                hypothesis_embedding = torch.matmul(W_star.unsqueeze(0), k_source_embeddings).squeeze(0) # Shape [embed_dim]

                # Calculate L1 distance between hypothesis and predicted embedding (Eq. 9) 
                loss_j = F.l1_loss(target_embeddings[j], hypothesis_embedding, reduction='sum') # Using sum as L1 distance 'd'

                total_loss += loss_j
                valid_samples += 1

    if valid_samples == 0:
        return torch.tensor(0.0, device=target_embeddings.device) # Avoid division by zero

    return total_loss / valid_samples # Average loss over valid samples 


def gaze_loss_cosine(gaze_pred, gaze_gt):
    """
    Calculates the gaze estimation loss using cosine similarity (Eq. 10).

    Args:
        gaze_pred (Tensor): Predicted gaze vectors (shape: [B, gaze_dim]).
        gaze_gt (Tensor): Ground truth gaze vectors (shape: [B, gaze_dim]).

    Returns:
        Tensor: Scalar loss value (mean angular error in radians).
    """
    # Normalize vectors to ensure they are unit vectors for angular calculation
    gaze_pred_norm = F.normalize(gaze_pred, p=2, dim=1)
    gaze_gt_norm = F.normalize(gaze_gt, p=2, dim=1)

    # Calculate cosine similarity (dot product of unit vectors)
    cos_sim = torch.sum(gaze_pred_norm * gaze_gt_norm, dim=1)

    # Clamp values to avoid numerical issues with acos outside [-1, 1]
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)

    # Calculate angular difference in radians 
    angle_diff_rad = torch.acos(cos_sim)

    # Return the mean angle difference
    return torch.mean(angle_diff_rad)


# --- Example Usage (Conceptual Training Step) ---
if __name__ == '__main__':
    # Dummy Data
    Bs = 64 # Source batch size 
    Bt = 64 # Target batch size 
    img_size = (64, 256) # Example input image size based on paper (h, w)
    gaze_dim = 2

    source_images = torch.randn(Bs, 3, img_size[0], img_size[1])
    source_gaze_gt = F.normalize(torch.randn(Bs, gaze_dim), p=2, dim=1) # Example unit gaze vectors
    target_images = torch.randn(Bt, 3, img_size[0], img_size[1])

    # Model
    model = DAGEN(embedding_dim=EMBEDDING_DIM, gaze_dim=gaze_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4) # 

    # --- Pre-training Step (Conceptual) ---
    print("--- Simulating Pre-training ---")
    model.train()
    optimizer.zero_grad()
    source_embeddings_pre, source_gaze_pred_pre = model(source_images)
    loss_gaze_pre = gaze_loss_cosine(source_gaze_pred_pre, source_gaze_gt)
    loss_gaze_pre.backward()
    optimizer.step()
    print(f"Pre-training Gaze Loss: {loss_gaze_pre.item():.4f}")


    # --- Joint Optimization Step (Conceptual) ---
    print("\n--- Simulating Joint Optimization Step ---")
    model.train()
    optimizer.zero_grad()

    # Forward pass for both domains
    source_embeddings, source_gaze_pred = model(source_images)
    target_embeddings, target_gaze_pred = model(target_images) # Target predictions needed for LLR neighbors 

    # Calculate losses
    # 1. Gaze loss on source domain 
    loss_gaze = gaze_loss_cosine(source_gaze_pred, source_gaze_gt)

    # 2. EPC loss between target and source 
    # Note: Using source_gaze_gt for LLR as mentioned in paper for higher accuracy 
    loss_epc = epc_loss(target_embeddings, target_gaze_pred,
                        source_embeddings, source_gaze_gt, # Using source GT here
                        k=K_NEIGHBORS, lambda_reg=LAMBDA_REG, mu_select=MU_SELECT)

    # 3. Total DA loss (Eq. 8) 
    total_loss = (LAMBDA_GAZE * loss_gaze) + (LAMBDA_EPC * loss_epc)

    # Backpropagation
    total_loss.backward()
    optimizer.step()

    print(f"Joint Gaze Loss: {loss_gaze.item():.4f}")
    print(f"Joint EPC Loss: {loss_epc.item():.4f}")
    print(f"Total Joint Loss: {total_loss.item():.4f}")

