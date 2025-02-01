import os
import torch
from torch.nn.functional import cross_entropy
from tqdm import tqdm
import wandb

from data import DNADataset
from models import VAE


DEVICE = torch.device("cuda:2")

train_config = {
    "learning_rate": 1e-4,
    "epochs": 100,
    "cycles": 10,
    "batch_size": 64,
}
model_config = {
    "encoded_dim": 64,
    "hidden_dim": 64,
    "stages": 6,
    "blocks": 3,
}

if __name__ == "__main__":
    wandb.init(project="my-awesome-project", config={**train_config, **model_config})

    # Initialize the dataset and dataloader
    dataset = DNADataset("hf://datasets/Zehui127127/latent-dna-diffusion/sequence.csv")
    dataloader = torch.utils.data.DataLoader(
        dataset, train_config["batch_size"], shuffle=True
    )
    steps_each_cycle = len(dataloader) * (
        train_config["epochs"] // train_config["cycles"]
    )

    # Initialize the model and trainer
    model = VAE(input_dim=5, cond_dim=15, **model_config)
    ckpt_dir = "src/models/vae/ckpt/"
    torch.save(model, ckpt_dir + f"vae_epoch_{0}.pth")

    # Training loop
    model = model.to(device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["learning_rate"])
    model.train()
    step = 0
    for epoch in range(train_config["epochs"]):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Compression {model.compression}")
        for i, (x, c) in enumerate(pbar):
            # Update beta
            step = (step + 1) % steps_each_cycle
            beta = min(1.0, 2 * step / steps_each_cycle)
            wandb.log({"beta": beta})

            # Forward pass
            x = x.to(device=DEVICE, non_blocking=True)
            c = c.to(device=DEVICE, non_blocking=True)
            mu, sigma = model.encode(x, c)
            z = mu + sigma * torch.randn_like(mu)
            x_recon = model.decode(z, c)

            # Loss
            loss_recon = cross_entropy(x_recon.transpose(-1, -2), x.transpose(-1, -2))
            factor = 0.5 * (mu.shape[-1] * mu.shape[-2]) / x.shape[-2]
            loss_kl = factor * (sigma**2 + mu**2 - (sigma**2).log() - 1).mean()
            loss = loss_recon + beta * loss_kl

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Logging
            wandb.log({"loss_recon": loss_recon.item()})
            wandb.log({"loss_kl": loss_kl.item()})
            wandb.log({"loss_elbo": (loss_recon + loss_kl).item()})
            pbar.set_postfix(
                recon=f"{loss_recon.item():.3f}", kl=f"{loss_kl.item():.3f}"
            )

        if (epoch + 1) % 10 == 0:
            torch.save(model, ckpt_dir + f"vae_epoch_{epoch+1}.pth")
