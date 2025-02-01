import torch
from torch.nn.functional import cross_entropy
from tqdm import tqdm
import wandb

from data import DNADataset
from vae import VAE


DEVICE = torch.device("cuda:0")
CKPT_DIR = "src/models/vae/ckpt/"

train_config = {
    "learning_rate": 3e-4,
    "epochs": 100,
    "cycles": 5,
    "batch_size": 256,
    "max_beta": 0.5,
}

model_config = {
    "input_dim": 5,
    "encoded_dim": 64,
    "blocks": 4,
}

if __name__ == "__main__":
    wandb.init(project="my-awesome-project", config={**train_config, **model_config})

    # Initialize the dataset and dataloader
    dataset = DNADataset("hf://datasets/Zehui127127/latent-dna-diffusion/sequence.csv")
    dataloader = torch.utils.data.DataLoader(
        dataset, train_config["batch_size"], shuffle=True, num_workers=1
    )

    # Initialize the model and trainer
    model = VAE(**model_config)
    torch.save(model, CKPT_DIR + f"vae_epoch_{0}.pth")

    # Training loop
    model = model.to(device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["learning_rate"])
    model.train()
    step = 0
    for epoch in range(train_config["epochs"]):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Compression {model.compression}")
        steps_each_cycle = len(dataloader) * (
            train_config["epochs"] // train_config["cycles"]
        )
        for i, (x, c) in enumerate(pbar):
            # Update beta
            step = (step + 1) % steps_each_cycle
            beta = train_config["max_beta"] * min(1.0, 2 * step / steps_each_cycle)

            # Forward pass
            x = x.transpose(-1, -2).to(device=DEVICE, non_blocking=True)
            mu, sigma = model.encode(x)
            z = mu + sigma * torch.randn_like(mu)
            x_recon = model.decode(z)
            x_recon = torch.tanh(x_recon / 10) * 10  # soft clip to (-10, 10)

            # Loss
            loss_recon = cross_entropy(x_recon, x)
            factor = 0.5 * (mu.shape[-1] * mu.shape[-2]) / x.shape[-1]
            loss_kl = factor * (sigma**2 + mu**2 - (sigma**2).log() - 1).mean()
            loss = loss_recon + beta * loss_kl

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # Logging
            pbar.set_postfix(
                recon=f"{loss_recon.item():.4f}", kl=f"{loss_kl.item():.4f}"
            )
            wandb.log(
                {
                    "beta": beta,
                    "loss_recon": loss_recon.item(),
                    "loss_kl": loss_kl.item(),
                    "loss_elbo": (loss_recon + loss_kl).item(),
                }
            )

        if (epoch + 1) % 10 == 0:
            torch.save(model, CKPT_DIR + f"vae_epoch_{epoch+1}.pth")
