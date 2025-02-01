import torch
from torch.nn.functional import cross_entropy
from tqdm import tqdm
import wandb

from data import DNADataset
from vae import VQVAE


DEVICE = torch.device("cuda:2")
CKPT_DIR = "src/models/vae/ckpt/"

train_config = {
    "learning_rate": 3e-4,
    "epochs": 100,
    "batch_size": 256,
}

model_config = {
    "input_dim": 5,
    "encoded_dim": 64,
    "blocks": 4,
    "fsq_levels": 5,
}

if __name__ == "__main__":
    wandb.init(project="my-awesome-project", config={**train_config, **model_config})

    # Initialize the dataset and dataloader
    dataset = DNADataset("hf://datasets/Zehui127127/latent-dna-diffusion/sequence.csv")
    dataloader = torch.utils.data.DataLoader(
        dataset, train_config["batch_size"], shuffle=True, num_workers=1
    )

    # Initialize the model and trainer
    model = VQVAE(**model_config)
    torch.save(model, CKPT_DIR + f"vqvae_epoch_{0}.pth")

    # Training loop
    model = model.to(device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["learning_rate"])
    model.train()
    for epoch in range(train_config["epochs"]):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Compression {model.compression}")
        for i, (x, c) in enumerate(pbar):
            # Forward pass
            x = x.transpose(-1, -2).to(device=DEVICE, non_blocking=True)
            z = model.encode(x)
            x_recon = model.decode(z)
            x_recon = torch.tanh(x_recon / 10) * 10  # soft clip to (-10, 10)
            loss_recon = cross_entropy(x_recon, x)

            # Backward pass
            optimizer.zero_grad()
            loss_recon.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # Logging
            wandb.log({"loss_recon": loss_recon.item()})
            pbar.set_postfix(recon=f"{loss_recon.item():.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model, CKPT_DIR + f"vqvae_epoch_{epoch+1}.pth")
