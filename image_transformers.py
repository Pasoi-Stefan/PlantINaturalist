import torchvision.transforms as transforms


class transform:
    @staticmethod
    def get():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

class transform_autoaugment:
    @staticmethod
    def get():
        return transforms.Compose([
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])