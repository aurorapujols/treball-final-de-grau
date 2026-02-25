import torchvision.transforms as T

base_transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])