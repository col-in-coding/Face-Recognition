import torch
import torch.nn as nn

if __name__ == "__main__":
    inp = torch.tensor([[0.2, 0.8]])
    pred = nn.Softmax(dim=1)(inp)
    print(pred)
    log = torch.log(pred)
    print(log)
    label = torch.tensor([0])
    nll_loss = nn.NLLLoss()
    print("My Loss: ", nll_loss(log, label))

    criterion = nn.CrossEntropyLoss()
    loss = criterion(inp, label)
    print("\nCrossEntropy: ", loss)

    l2n = torch.norm(inp)
    print(l2n)
