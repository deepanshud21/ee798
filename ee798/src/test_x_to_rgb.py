import torch
from torch.utils.data import DataLoader
from models import XtoRGBModel
from dataset import IntrinsicDataset
from torchvision import transforms

def test_x_to_rgb():
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    intrinsic_paths = [...]
    image_paths = [...]
    
    dataset = IntrinsicDataset(intrinsic_paths, image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = XtoRGBModel()
    model.load_state_dict(torch.load('x_to_rgb_model.pth'))
    model.eval()

    total_loss = 0
    criterion = torch.nn.MSELoss()
    
    for x, rgb in dataloader:
        with torch.no_grad():
            output = model(x)
            loss = criterion(output, rgb)
            total_loss += loss.item()
    print(f'Average Loss: {total_loss / len(dataloader)}')

if __name__ == "__main__":
    test_x_to_rgb()
