# -*- coding: utf-8 -*-
# @Time    : 2020-04-14 14:32
# @Author  : daijianhao
import torch.nn as nn
from torchvision import transforms, datasets,models
import json
import os
import torch.optim as optim

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "eval": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

dataPath = "/Users/djj/Desktop/smoke_recognition-master/data/"
path_tranin = dataPath + "image_train"
path_eval = dataPath + "image_eval"


train_dataset = datasets.ImageFolder(root=path_tranin,
                                     transform=data_transform["train"])
train_num = len(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=32, shuffle=True,
                                           num_workers=0)

eval_dataset = datasets.ImageFolder(root=path_eval,
                                        transform=data_transform["eval"])
eval_num = len(eval_dataset)
eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=32, shuffle=False,
                                              num_workers=0)
model_name = "vgg"
net = models.vgg16(pretrained=True)

#net = vgg(model_name=model_name, class_num=5, init_weights=True)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

best_acc = 0.0
save_path = './{}Net.pth'.format(model_name)

for epoch in range(30):
    net.train()
    running_loss = 0.0
    step = 0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()


    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():

        for data_test in eval_loader:
            test_images, test_labels = data_test
            optimizer.zero_grad()
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()

        accurate_test = acc / eval_num

        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)

        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, accurate_test))

print('Finished Training')