import torch
from diffusers import StableDiffusionPipeline
import cv2
import matplotlib.pyplot as plt

MODEL_DIR = "checkpoints/rgb2x_epoch_10"

def visualize_aov(input_image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StableDiffusionPipeline.from_pretrained(MODEL_DIR).to(device)

    input_image = cv2.imread(input_image_path)
    input_tensor = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output_aov = model(input_tensor)

    output_image = output_aov.cpu().squeeze().numpy().transpose(1, 2, 0)
    plt.imshow(output_image)
    plt.title("Generated AOV")
    plt.show()

if __name__ == "__main__":
    visualize_aov("data/sample_image.png")
