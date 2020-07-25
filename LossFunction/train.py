import logging
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim

from customDatasets import CASIAWebFace
from SEResNet_IR import SEResNet34_IR
from MarginLayer import InnerProduct, CosineMarginProduct

import pdb

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters


if __name__ == "__main__":
    feature_dim = 512
    margin_type = "InnerProduct"

    # dataset loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_root = \
        "/home/coin/Documents/face_recognition/hw5/data/"
    file_list = "/home/coin/Documents/face_recognition/hw5/data_list.txt"
    trainset = CASIAWebFace(train_root, file_list, transform=transform)
    trainloader = DataLoader(trainset, batch_size=2, shuffle=True,
                             num_workers=1, drop_last=False)
    net = SEResNet34_IR().to(device)

    # define margin layer
    if margin_type == "InnerProduct":
        margin = InnerProduct(feature_dim, trainset.class_nums).to(device)
    elif margin_type == "CosFace":
        margin = CosineMarginProduct(feature_dim, trainset.class_nums)
    else:
        logging.info(margin_type, " is not available")

    # define optimizers for different layer
    criterion_classi = torch.nn.CrossEntropyLoss().to(device)
    optimizer_classi = optim.SGD([
        {"params": net.parameters(), "weight_decay": 5e-4},
        {"params": margin.parameters(), "weight_decay": 5e-4}
    ], lr=0.1, momentum=0.9, nesterov=True)
    scheduler_classi = optim.lr_scheduler.MultiStepLR(optimizer_classi,
                                                      milestones=[20, 35, 45],
                                                      gamma=0.1)

    epochs = 100
    losses = []
    for epoch in range(1, epochs + 1):
        # train model
        logging.info("Train Epoch: {}/{}".format(epoch, epochs))
        net.train()

        for batch in trainloader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            feature = net(imgs)
            output = margin(feature)
            print(output.shape)
            # print(labels)
            loss = criterion_classi(output, labels)
            print(loss)
            pdb.set_trace()
            losses.append(loss)
            optimizer_classi.zero_grad()
            loss.backward()
            optimizer_classi.step()
        print("loss: ", loss)
        pdb.set_trace()
        scheduler_classi.step()
