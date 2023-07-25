from pathlib import Path
from typing import Any, Callable, Optional, Tuple
import PIL.Image
from torch.utils.data import Dataset
from scipy.io import loadmat
import torch
from torchvision import transforms, models
from collections import OrderedDict
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt


def find_classes(labels):
    classes = sorted(list(set(labels)))
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class MyDataSet(Dataset):
    def __init__(self, split: str = "train", transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:

        self._base_folder = "/Users/ahmadessa/Downloads/flowers-102"
        self._images_folder = "/Users/ahmadessa/Downloads/flowers-102/jpg"
        self.transform = transform
        self.target_transform = target_transform

        set_ids = loadmat("/Users/ahmadessa/Downloads/flowers-102/setid.mat", squeeze_me=True)

        _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}
        image_ids = set_ids[_splits_map[split]].tolist()

        labels = loadmat("/Users/ahmadessa/Downloads/flowers-102/imagelabels.mat", squeeze_me=True)
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))

        classes, class_to_idx = find_classes(labels)

        self.classes = classes
        self.class_to_idx = class_to_idx

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(Path(self._images_folder) / f"image_{image_id:05d}.jpg")

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def load_images():
    # 6149 / 143 , 1020/60
    means_norm = [0.485, 0.456, 0.406]
    standard_deviations_norm = [0.229, 0.224, 0.225]
    model_input_size = 224
    batch_s = 64

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(model_input_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(30),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means_norm,
                                                                standard_deviations_norm)])
    test_and_validation_transforms = [transforms.Resize(256),
                                          transforms.CenterCrop(model_input_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means_norm,
                                                               standard_deviations_norm)]

    validation_transforms = transforms.Compose(test_and_validation_transforms)

    test_transforms = transforms.Compose(test_and_validation_transforms)

    train_data = MyDataSet(split="test", transform=train_transforms)
    valid_data = MyDataSet(split="val", transform=validation_transforms)
    test_data = MyDataSet(split="train", transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=143, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=60)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=60)

    return train_loader, test_loader, valid_loader, train_data, test_data, valid_data


def train_model(model, epochs, loss_function, opt_func, training_loader, validation_loader):
    model.train()
    epoch_train_loss = []
    epoch_valid_loss = []
    epoch_valid_acc = []

    for epoch_index in range(epochs):
        running_loss = 0
        iter_num = 0
        for images, labels in iter(training_loader):
            iter_num += 1
            opt_func.zero_grad()
            output = model.forward(images)

            loss = loss_function(output, labels)
            loss.backward()
            opt_func.step()
            running_loss += loss.item()

        model.eval()
        validation_loss, accuracy = validate(model, loss_function, validation_loader)
        model.train()
        running_loss = running_loss / len(training_loader)

        epoch_train_loss.append(running_loss)
        epoch_valid_loss.append(validation_loss)
        epoch_valid_acc.append(accuracy)

        print("Epoch num: {}/{}".format(epoch_index + 1, epochs))
        print("Training Cross Entropy Loss: {:.2f} ".format(running_loss))
        print("Validation Cross Entropy Loss: {:.2f} ".format(validation_loss))
        print("Validation Accuracy: {:.2f}".format(accuracy))

    return epoch_train_loss, epoch_valid_loss, epoch_valid_acc


def validate(model, loss_function, validation_loader):
    valid_accuracy = 0
    valid_loss = 0

    for images, labels in iter(validation_loader):
        output = model.forward(images)
        valid_loss += loss_function(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
        valid_accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss / len(validation_loader), valid_accuracy / len(validation_loader)


def test_model(model, loss_function, test_loader):
    test_accuracy = 0
    test_loss = 0
    iter_num = 0

    accumulating_test_loss_list = []
    accumulating_test_acc_list = []

    for images, labels in iter(test_loader):
        iter_num += 1
        output = model.forward(images)
        test_loss += loss_function(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
        test_accuracy += equality.type(torch.FloatTensor).mean()

        accumulating_test_loss = test_loss / iter_num
        accumulating_test_acc = test_accuracy / iter_num

        accumulating_test_loss_list.append(accumulating_test_loss)
        accumulating_test_acc_list.append(accumulating_test_acc)

        print("Accumulating Test Loss: {:.2f} ".format(accumulating_test_loss))
        print("Accumulating Test Accuracy: {:.2f}".format(accumulating_test_acc))

    return accumulating_test_loss_list, accumulating_test_acc_list


train_loader1, test_loader1, valid_loader1, train_data1, test_data1, valid_data1 = load_images()

#model = models.vgg16(pretrained=True)
model = models.vgg19(pretrained=True)

for parameter in model.parameters():
        parameter.requires_grad = False

new_classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model.classifier = new_classifier

epoch_train_loss, epoch_valid_loss, epoch_valid_acc = train_model(
    model=model, epochs=10, loss_function=nn.CrossEntropyLoss(),
    opt_func=optim.Adam(model.classifier.parameters(), lr=0.001),
    training_loader=train_loader1, validation_loader=valid_loader1)


model.class_to_idx = test_data1.class_to_idx
accumulating_test_loss_list, accumulating_test_acc_list = test_model(model, nn.CrossEntropyLoss(), test_loader1)













plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc)
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()

plt.plot(range(len(epoch_train_loss)), epoch_train_loss,
         label='Training')
plt.plot(range(len(epoch_valid_loss)), epoch_valid_loss,
         label='Validation')
plt.ylabel('Cross Entropy Loss')
plt.xlabel('Epochs')
plt.legend(loc='lower right')
plt.show()

plt.plot(range(len(accumulating_test_loss_list)), accumulating_test_loss_list)
plt.ylabel('Accumulating Test Loss')
plt.xlabel('Batch num')
plt.show()


plt.plot(range(len(accumulating_test_acc_list)), accumulating_test_acc_list)
plt.ylabel('Accumulating Test Accuracy')
plt.xlabel('Batch num')
plt.show()





