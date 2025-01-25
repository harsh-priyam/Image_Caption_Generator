import torch 
import torchvision.transforms as transforms
from PIL import Image 

def print_examples(model,device,dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299,299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ]
    )

    model.eval()

    test_img1 = transform(Image.open("test_images/dog.jpg").convert("RGB")).unsqueeze(0)
    print("EXAMPLE 1 Correct: Dog walking in a green feild")
    print(
        "Example 1 Output: "
        + " ".join(model.caption_image(test_img1.to(device),dataset.vocab))
    )

    test_img2 = transform(
        Image.open("test_images/boat.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: A boat in the sea")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
    )

    test_img3 = transform(Image.open("test_images/car_street.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 3 CORRECT: Multiple cars of black color on street")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
    )
    model.train()

def save_checkpoint(state,filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step