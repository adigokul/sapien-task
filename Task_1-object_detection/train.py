import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SyntheticDataset, collate_fn
from model import SimpleDetector, DetectionLoss

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")

    data_dir = './data'
    num_classes = 5
    img_size = 224
    batch_size = 16
    num_epochs = 30
    lr = 0.001

    print("loading datasets...")
    train_dataset = SyntheticDataset(data_dir, num_samples=800, img_size=img_size, mode='train')
    val_dataset = SyntheticDataset(data_dir, num_samples=200, img_size=img_size, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")

    model = SimpleDetector(num_classes=num_classes, img_size=img_size)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {total_params:,}")

    criterion = DetectionLoss(num_classes=num_classes, img_size=img_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_loss = float('inf')
    train_losses = []
    val_losses = []

    print("\nstarting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            for t in targets:
                t['boxes'] = t['boxes'].to(device)
                t['labels'] = t['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                for t in targets:
                    t['boxes'] = t['boxes'].to(device)
                    t['labels'] = t['labels'].to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        elapsed = time.time() - start_time
        print(f"epoch {epoch+1}/{num_epochs} | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f} | time: {elapsed:.1f}s")

        scheduler.step()

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, './checkpoints/best_model.pth')
            print(f"  saved best model!")

    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss,
    }, './checkpoints/final_model.pth')

    with open('./outputs/training_log.txt', 'w') as f:
        f.write("epoch,train_loss,val_loss\n")
        for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
            f.write(f"{i+1},{tl:.4f},{vl:.4f}\n")

    print("\ntraining done!")
    print(f"best val loss: {best_loss:.4f}")

    return model, train_losses, val_losses

if __name__ == '__main__':
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./outputs', exist_ok=True)
    train()
