"""
dist_model.py — DistogramNet for CSOC-SSC
Trains a Multi-task Neural Network to predict 36-bin distograms from sequence features.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + res)

class DistogramNet(nn.Module):
    def __init__(self, seq_len=192, embed_dim=64):
        super().__init__()
        # 1D Sequence Processing
        self.embed = nn.Embedding(21, embed_dim)
        self.res_blocks = nn.Sequential(*[ResidualBlock1D(embed_dim) for _ in range(4)])
        
        # 2D Pairwise Processing
        self.conv2d_1 = nn.Conv2d(embed_dim*2, 128, kernel_size=3, padding=1)
        self.res2d = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Multi-task Heads
        self.dist_head = nn.Conv2d(128, 36, kernel_size=1) # 36 bins (2-20A)
        self.contact_head = nn.Conv2d(128, 1, kernel_size=1)
        self.q3_head = nn.Conv1d(embed_dim, 3, kernel_size=1)

    def forward(self, x):
        # x shape: (B, L)
        e = self.embed(x).transpose(1, 2) # (B, C, L)
        e = self.res_blocks(e)
        
        # Outer concatenation for 2D map
        B, C, L = e.shape
        e_expand1 = e.unsqueeze(3).expand(B, C, L, L)
        e_expand2 = e.unsqueeze(2).expand(B, C, L, L)
        pair_rep = torch.cat([e_expand1, e_expand2], dim=1) # (B, 2C, L, L)
        
        pair_rep = torch.relu(self.conv2d_1(pair_rep))
        pair_rep = self.res2d(pair_rep) + pair_rep
        
        # Predictions
        distogram = self.dist_head(pair_rep) # (B, 36, L, L)
        contact = torch.sigmoid(self.contact_head(pair_rep)) # (B, 1, L, L)
        q3 = self.q3_head(e) # (B, 3, L)
        
        return distogram, contact, q3

def train(args):
    print(f"Initializing DistogramNet Training on {args.epochs} epochs...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistogramNet(crop=args.crop).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    os.makedirs(args.out, exist_ok=True)
    print(f"Using device: {device}. Ready for training loop over {args.pdb_dir}")
    
    # Dummy Training Loop for integration
    for ep in range(args.epochs):
        # In actual execution, load Dataloader here
        loss = torch.tensor(0.5, requires_grad=True) 
        loss.backward()
        optimizer.step()
        
        if (ep+1) % 10 == 0:
            torch.save(model.state_dict(), f"{args.out}/distnet_ep{ep+1}.pt")
            print(f"Epoch {ep+1}/{args.epochs} | Loss: {loss.item():.4f} | Saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--pdb_dir', type=str, required=True)
    parser.add_argument('--labels_csv', type=str, default='')
    parser.add_argument('--out', type=str, default='checkpoints/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--crop', type=int, default=192)
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
