import torch 
import torchvision.transforms as transforms
from PIL import Image
from src.model import CNNtoRNN
from src.data_loader import get_loader

def setup_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transformation
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load dataset and data loader
    train_loader, dataset = get_loader(
        root_folder="Images",
        annotation_file="captions.txt",
        transform=transform,
        num_workers=2
    )
    
    # Model parameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    
    # Initialize model
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    
    # Load checkpoint properly
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, device, dataset, transform

def generate_caption(image_path, model, device, dataset, transform):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        res = []
        
        # Generate caption
        with torch.no_grad():
            caption = model.caption_image(image_tensor, dataset.vocab)
            for i in caption:
                if i == "<SOS>" or i == "<EOS>" or i == "<eos>" or i == "<sos>":
                    continue
                else:
                    res.append(i)
        
        return " ".join(res)
        
    except FileNotFoundError:
        return f"Error: Image file {image_path} not found."
    except Exception as e:
        return f"Error generating caption: {str(e)}"

def main():
    # Setup model and dependencies
    model, device, dataset, transform = setup_model("final_model.pth")
    
    # Generate caption for test image
    test_img_path = "test_images/girl.jpg"
    caption = generate_caption(test_img_path, model, device, dataset, transform)
    print(f"Generated caption: {caption}")

if __name__ == "__main__":
    main()