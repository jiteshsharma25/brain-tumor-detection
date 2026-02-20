import torch
from torch.utils.data import DataLoader
from utils.dataset import BrainDataset
from scripts.model import BrainTumorModel

data_dir = r"C:\Users\sgari\Downloads\aimodel\data\processed"

dataset = BrainDataset(data_dir)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = BrainTumorModel()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    total_loss = 0

    for i, (images, labels) in enumerate(loader):
        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # print every 10 steps
        if i % 10 == 0:
            print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item()}")

    print(f"Epoch {epoch+1} DONE, Total Loss: {total_loss}")
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")