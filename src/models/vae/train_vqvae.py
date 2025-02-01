import torch
from torch.nn.functional import cross_entropy
from tqdm import tqdm
import wandb

from data import DNADataset
from models import VQVAE


DEVICE = torch.device("cuda:0")

train_config = {
    "learning_rate": 1e-4,
    "epochs": 100,
    "batch_size": 64,
}
model_config = {
    "encoded_dim": 64,
    "hidden_dim": 64,
    "stages": 6,
    "blocks": 3,
    "fsq_levels": 2,
}

if __name__ == "__main__":
    wandb.init(project="my-awesome-project", config={**train_config, **model_config})

    # Initialize the dataset and dataloader
    dataset = DNADataset("hf://datasets/Zehui127127/latent-dna-diffusion/sequence.csv")
    dataloader = torch.utils.data.DataLoader(
        dataset, train_config["batch_size"], shuffle=True
    )

    # Initialize the model and trainer
    model = VQVAE(input_dim=5, cond_dim=15, **model_config)
    ckpt_dir = "src/models/vae/ckpt/"
    model_name = "".join([f"{k}={v}_" for k, v in model_config.items()])
    torch.save(model, ckpt_dir + f"vqvae_{model_name}epoch={0}.pth")

    # Training loop
    model = model.to(device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["learning_rate"])
    model.train()
    for epoch in range(train_config["epochs"]):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Compression {model.compression}")
        for i, (x, c) in enumerate(pbar):
            # Forward pass
            x = x.to(device=DEVICE, non_blocking=True)
            c = c.to(device=DEVICE, non_blocking=True)
            z = model.encode(x, c)
            x_recon = model.decode(z, c)
            loss = cross_entropy(x_recon.transpose(-1, -2), x.transpose(-1, -2))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Logging
            wandb.log({"loss_recon": loss.item()})
            pbar.set_postfix(recon=f"{loss.item():.3f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model, ckpt_dir + f"vqvae_{model_name}epoch={epoch+1}.pth")
