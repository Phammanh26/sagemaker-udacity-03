# TODO: Import your dependencies.
# For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torchvision.models import resnet50, ResNet50_Weights
import os
import argparse
from torch.utils.data import Subset
import smdebug.pytorch as smd



from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(774400, 128)
        self.fc2 = nn.Linear(128, 133)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    
def test(model, test_loader, device, loss_fn, hook = None):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    
            
    model.eval()
    
    # if hook:
    #     hook.set_mode(smd.modes.EVAL)
        
    # ===================================================#
    # 2. Set the SMDebug hook for the validation phase. #
    # ===================================================#
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(
                test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def train(model, train_loader, epochs, loss_fn, optimizer, hook = None):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    model.train()
    print("START TRAINING")
    # if hook:
    #     hook.set_mode(smd.modes.TRAIN)
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to('cpu'), target.to('cpu')
            optimizer.zero_grad()
            output = model(data)
            # print(output.shape)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:

                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\t loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    
    return model

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = Net()
    
    return model


def model_fn(model_dir):
    model = net()
    print(f'Load trained model: {os.path.join(model_dir, "model.pt")}')
    with open(os.path.join(model_dir, "model.pt"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model


def create_data_loaders(data, batch_size):
    train_dataloader = DataLoader(
        data['train'], batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(
        data['validate'], batch_size=batch_size, shuffle=True)
    return train_dataloader, valid_dataloader


def _create_datasets(root_path, num_samples=None):
    # Define the transformations to apply to the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the images to a specific size
        transforms.ToTensor(),  # Convert the images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])  # Normalize the image tensors
    ])

    dataset = datasets.ImageFolder(root_path, transform=transform)
    
    if num_samples:
    
        # Create a Subset containing only the first num_samples samples
        subset_indices = list(range(num_samples))
        subset_dataset = Subset(dataset, subset_indices)
    else:
        subset_dataset = dataset
    return subset_dataset


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model = net()
    # Setup hook 
    hook = None #smd.Hook.create_from_json_file()
    # hook.register_hook(model)

    '''
    TODO: Create your loss and optimizer
    '''
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    
        
    # Create data loader
    print("Load and Prepare dataset")
    data = {}
    data['train'] = _create_datasets(root_path=args.train, num_samples = args.num_samples)
    data['validate'] = _create_datasets(root_path=args.validate, num_samples = args.num_samples)
    
    train_loader, test_loader = create_data_loaders(
        data, batch_size=args.batch_size)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    
    
    print("Begin training")

        
    model = train(model=model, 
                  epochs = args.epochs,
                  train_loader=train_loader,
                  loss_fn=loss_fn, 
                  optimizer=optimizer,
                 hook = hook)

    '''
    TODO: Test the model to see its accuracy
    '''
    test(model=model, 
         test_loader=test_loader,
         device=args.device, 
         loss_fn=loss_fn,
        hook = hook)

    '''
    TODO: Save the trained model
    '''
    
    print("Begin save model")
    torch.save(model, os.path.join(args.model_dir,"model.pt"))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    parser.add_argument(
        "--batch-size", type=int, default=64, help="batch size"
    )

    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )

    parser.add_argument(
        "--log-interval", type=int, default=10, help="how many batches to wait before logging training status"
    )

    parser.add_argument(
        "--device", type=str, default="cpu" if torch.cuda.is_available() else "cpu", help="device (cuda or cpu)"
    )

    parser.add_argument(
        "--train", type=str, default=os.getenv('SM_CHANNEL_TRAIN'), help="path to training data"
    )

    parser.add_argument(
        "--validate", type=str, default=os.getenv('SM_CHANNEL_VALIDATE'), help="path to validation data"
    )

    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )

    parser.add_argument(
        "--model-dir", type=str, default=os.getenv('SM_MODEL_DIR') , help="dir to save the trained model"
    )
    
    parser.add_argument(
        "--num-samples", type=int, default=100, help="number sample data used for train and test"
    )
    
    
    

    args = parser.parse_args()

    main(args)
