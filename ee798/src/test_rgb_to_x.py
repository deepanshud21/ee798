import torch
from torch.utils.data import DataLoader
from models import RGBtoXModel
from dataset import IntrinsicDataset
from torchvision import transforms

def test_rgb_to_x():
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    intrinsic_paths = [...]
    image_paths = [...]
    
    dataset = IntrinsicDataset(intrinsic_paths, image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = RGBtoXModel()
    model.load_state_dict(torch.load('rgb_to_x_model.pth'))
    model.eval()

    total_loss = 0
    criterion = torch.nn.MSELoss()
    
    for rgb, target in dataloader:
        with torch.no_grad():
            output = model(rgb)
            loss = criterion(output, target)
            total_loss += loss.item()
    print(f'Average Loss: {total_loss / len(dataloader)}')

if __name__ == "__main__":
    test_rgb_to_x()
