import torch
from torch.utils.data import DataLoader
from models import XtoRGBModel
from dataset import IntrinsicDataset
from torchvision import transforms

def train_x_to_rgb():
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    intrinsic_paths = [...]
    image_paths = [...]
    
    dataset = IntrinsicDataset(intrinsic_paths, image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = XtoRGBModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    for epoch in range(10):
        total_loss = 0
        for x, rgb in dataloader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, rgb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}: Loss = {total_loss / len(dataloader)}')

if __name__ == "__main__":
    train_x_to_rgb()
