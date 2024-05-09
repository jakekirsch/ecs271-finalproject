import torch
import torch.nn as nn
from utils.image_utils import load_dataset
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # left
        self.left_conv_1 = DoubleConv(1, 64)
        self.down_1 = nn.MaxPool2d(2, 2)

        self.left_conv_2 = DoubleConv(64, 128)
        self.down_2 = nn.MaxPool2d(2, 2)

        self.left_conv_3 = DoubleConv(128, 256)
        self.down_3 = nn.MaxPool2d(2, 2)

        self.left_conv_4 = DoubleConv(256, 512)
        self.down_4 = nn.MaxPool2d(2, 2)

        # center
        self.center_conv = DoubleConv(512, 1024)

        # right
        self.up_1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.right_conv_1 = DoubleConv(1024, 512)

        self.up_2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.right_conv_2 = DoubleConv(512, 256)

        self.up_3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.right_conv_3 = DoubleConv(256, 128)

        self.up_4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.right_conv_4 = DoubleConv(128, 64)

        # output
        self.output = nn.Conv2d(64, 5, 1, 1, 0)  # out_channels: 5

    def forward(self, x):
        # left
        x1 = self.left_conv_1(x)
        x1_down = self.down_1(x1)

        x2 = self.left_conv_2(x1_down)
        x2_down = self.down_2(x2)

        x3 = self.left_conv_3(x2_down)
        x3_down = self.down_3(x3)

        x4 = self.left_conv_4(x3_down)
        x4_down = self.down_4(x4)

        # center
        x5 = self.center_conv(x4_down)

        # right
        x6_up = self.up_1(x5)
        temp = torch.cat((x6_up, x4), dim=1)
        x6 = self.right_conv_1(temp)

        x7_up = self.up_2(x6)
        temp = torch.cat((x7_up, x3), dim=1)
        x7 = self.right_conv_2(temp)

        x8_up = self.up_3(x7)
        temp = torch.cat((x8_up, x2), dim=1)
        x8 = self.right_conv_3(temp)

        x9_up = self.up_4(x8)
        temp = torch.cat((x9_up, x1), dim=1)
        x9 = self.right_conv_4(temp)

        # output
        output = self.output(x9)

        return output
def iou_(outputs: torch.Tensor, labels: torch.Tensor):

    outputs = outputs.view(-1)
    labels = labels.view(-1)

    intersection = torch.logical_and(outputs, labels).sum()
    union = torch.logical_or(outputs, labels).sum()

    IoU = intersection.float() / union.float() if union != 0 else torch.tensor(float('nan'))
    IoU.requires_grad_()
    return IoU

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. ")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU instead.")

    image, label = load_dataset('../data/Patient_01.nii', '../data/GT.nii') #shape(512,512,229)
    tensor_x = torch.from_numpy(image).float()
    tensor_y = torch.from_numpy(label).float()
    X = tensor_x.permute(2, 0, 1).unsqueeze(1)  # x is now (229, 1, 512, 512)
    Y = tensor_y.permute(2, 0, 1).unsqueeze(1)

    #cropping function
    X = F.interpolate(X, size=(256, 256), mode='bilinear', align_corners=False)  # x is now (229, 1, 256, 256) (229, 256, 256)
    print(X.shape)
    Y = F.interpolate(Y, size=(256, 256), mode='bilinear', align_corners=False)
    print(X.shape)

    dataset = CustomDataset(X, Y)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = UNet()
    criterion = torch.nn.CrossEntropyLoss()  # or any other applicable loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5  # Number of epochs
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.cpu()
            #predictions = torch.argmax(outputs, dim=1).float()
            loss = criterion(outputs, targets.squeeze().long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss -= loss.item()
            inputs.cpu()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(data_loader)}')

    model.cpu()
