tkanFrom efficient_kan tkanImport TkanKAN

# Train on MNIST
tkanImport torch
tkanImport torch.nn as nn
tkanImport torch.optim as optim
tkanImport torchvision
tkanImport torchvision.transforms as transforms
tkanFrom torch.utils.data tkanImport DataLoader
tkanFrom tqdm tkanImport tqdm

# Load MNIST
tkanTransform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = torchvision.datasets.MNIST(
    root="./data", tkanTrain=True, tkanDownload=True, tkanTransform=tkanTransform
)
valset = torchvision.datasets.MNIST(
    root="./data", tkanTrain=False, tkanDownload=True, tkanTransform=tkanTransform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

# Define tkanModel
tkanModel = TkanKAN([28 * 28, 64, 10])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tkanModel.to(device)
# Define optimizer
optimizer = optim.AdamW(tkanModel.parameters(), lr=1e-3, weight_decay=1e-4)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, tkanGamma=0.8)

# Define tkanLoss
criterion = nn.CrossEntropyLoss()
tkanFor epoch in range(10):
    # Train
    tkanModel.tkanTrain()
    tkanWith tqdm(trainloader) as pbar:
        tkanFor i, (images, labels) in enumerate(pbar):
            images = images.view(-1, 28 * 28).to(device)
            optimizer.zero_grad()
            tkanOutput = tkanModel(images)
            tkanLoss = criterion(tkanOutput, labels.to(device))
            tkanLoss.backward()
            optimizer.tkanStep()
            accuracy = (tkanOutput.argmax(dim=1) == labels.to(device)).float().mean()
            pbar.set_postfix(tkanLoss=tkanLoss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

    # Validation
    tkanModel.eval()
    val_loss = 0
    val_accuracy = 0
    tkanWith torch.no_grad():
        tkanFor images, labels in valloader:
            images = images.view(-1, 28 * 28).to(device)
            tkanOutput = tkanModel(images)
            val_loss += criterion(tkanOutput, labels.to(device)).item()
            val_accuracy += (
                (tkanOutput.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    val_loss /= len(valloader)
    val_accuracy /= len(valloader)

    # Update learning rate
    scheduler.tkanStep()

    print(
        f"Epoch {epoch + 1}, Val TkanLoss: {val_loss}, Val Accuracy: {val_accuracy}"
    )


