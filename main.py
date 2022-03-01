import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
from itertools import product
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

from collections import OrderedDict
from collections import namedtuple
import math

import warnings
warnings.filterwarnings("ignore")


def downloadData(batch_size):
    """Get the Fashion MNIST data set and parse it"""
    global script_path
    script_path = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))

    train_set = torchvision.datasets.FashionMNIST(
        root=script_path + "/data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=1)

    # num_workers = 1 as it is proved that it is the best
    return train_set, train_loader


class Network(nn.Module):
    def __init__(self, train_set, train_loader, in_level, conv_level_1, conv_level_2, fc_level_1, fc_level_2, out_level,
                 batch_size, lr, shuffle):
        if isinstance(in_level, int) and isinstance(conv_level_1, int) and isinstance(conv_level_2, int) \
                and isinstance(fc_level_1, int) and isinstance(fc_level_2, int) and \
                isinstance(out_level, int):

            super(Network, self).__init__()
            self.train_set = train_set
            self.train_loader = train_loader

            self.conv1 = nn.Conv2d(in_channels=in_level, out_channels=conv_level_1, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=conv_level_1, out_channels=conv_level_2, kernel_size=5)
            self.fc1 = nn.Linear(in_features=conv_level_2*4*4, out_features=fc_level_1, bias=True)
            self.fc2 = nn.Linear(in_features=fc_level_1, out_features=fc_level_2, bias=True)
            self.out = nn.Linear(in_features=fc_level_2, out_features=out_level, bias=True)

            self.batch_size = batch_size
            self.lr = lr
            self.shuffle = shuffle

        else:
            raise ValueError("Network layers should have integer number for nodes")

    def forward(self, t):
        """Perform forward propagation"""
        t = F.relu((self.conv1(t))) # hidden conv layer 1
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.conv2(t)) # hidden conv layer 2
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.fc1(t.reshape(-1, 12*4*4)))  # hidden linear (fully connected) layer 1
        t = F.relu(self.fc2(t))  # hidden linear (fully connected) layer 2

        t = self.out(t)  # output layer
        # t = F.softmax(t)
        return t

    def normalizeData(self):

        num_pixels = len(self.train_set) * 28 * 28
        total_sum = 0
        for batch in self.train_loader:
            total_sum+=batch[0].sum()
        mean = total_sum / num_pixels

        sum_squared_error = 0
        for batch in self.train_loader:
            sum_squared_error+=((batch[0]-mean).pow(2).sum())
        std = torch.sqrt(sum_squared_error/num_pixels)

        train_set_normal = torchvision.datasets.FashionMNIST(
            root=script_path + "/data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
        )
        train_loader_normal = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                          shuffle=True, num_workers=1)
        self.train_loader = train_loader_normal
        self.train_set = train_set_normal

    def trainNetwork(self, epochs, comment="Default"):

        #self.normalizeData()

        optimizer = optim.Adam(network.parameters(), lr=self.lr)

        images, labels = next(iter(self.train_loader))
        grid = torchvision.utils.make_grid(images)

        #comment = f'batch_size={self.batch_size} lr={self.lr}'
        tb = SummaryWriter(comment)  # append the comment to the name of the specific run
        tb.add_image('images', grid)
        tb.add_graph(network, images)

        for epoch in range(epochs):
            total_loss = 0
            total_correct = 0
            all_preds = torch.tensor([])

            for batch in self.train_loader:
                images = batch[0].to('cuda')
                labels = batch[1].to('cuda')
                # images, labels = batch # cpu

                preds, loss, correct = self.predictBatch(images, labels, optimizer)
                all_preds = torch.cat((all_preds, preds), dim=0)
                total_loss += loss * self.batch_size
                total_correct += correct

            self.buildConfusionMatrix(all_preds)  # build confusion matrix

            print("Epoch {}: total_correct: {}, total_loss: {}, accuracy: {}".format(str(epoch),\
                    str(total_correct), str(total_loss), str(self.getAccuracy(total_correct, len(self.train_set)))))

            tb.add_scalar('Loss', total_loss, epoch)
            tb.add_scalar('Number correct', total_correct, epoch)
            tb.add_scalar('Accuracy', self.getAccuracy(total_correct, len(self.train_set)), epoch)

            for name, weight in network.named_parameters():
                tb.add_histogram(name, weight, epoch)
                tb.add_histogram(f'{name}.grad', weight.grad, epoch)

        tb.close()

    # @torch.no_grad()  # locally turn off gradient tracking
    def predictBatch(self, images, labels, optimizer):
        """Predict the labels of a batch of given images"""
        preds = self.forward(images)  # perform forward
        # .unsqueeze(0) for one image

        correct = self.getCorrect(preds, labels)
        loss = self.getLoss(preds, labels)

        optimizer.zero_grad()  # zero out the gradients
        loss.backward()  # perform backward i.e. calculate gradients
        optimizer.step()  # update weights

        return preds, loss.item(), correct

    def getCorrect(self, preds, labels):  # Count all correct predictions
        return preds.argmax(dim=1).eq(labels).sum().item()

    def getLoss(self, preds, labels):  # Calculates the loss
        return F.cross_entropy(preds, labels)

    def getAccuracy(self, total_correct, total_samples): # Calculates the accuracy
        return total_correct/total_samples

    def buildConfusionMatrix(self, preds):

        stacked = torch.stack(
            (
                self.train_set.targets,
                preds.argmax(dim=1)
            ), dim=1
        )  # stack pairs of actual label - prediction label

        cmt = torch.zeros(10, 10, dtype=torch.int32)  # initialise a confusion matrix
        for pair in stacked:

            val, pred = pair.tolist()
            cmt[val, pred] = cmt[val, pred] + 1

        fig, ax = plot_confusion_matrix(conf_mat=cmt.numpy(), colorbar=True, show_absolute=True,
                                        show_normed=True, class_names=self.train_set.classes,
                                        figsize=(15, 15))
        plt.show()


class RunBuilder:
    def __init__(self, lr_list, batch_list, shuffle_list, epochs):
        self.lr_list = lr_list
        self.batch_list = batch_list
        self.shuffle_list = shuffle_list
        self.epochs = epochs

        self.params = OrderedDict(
            lr=self.lr_list,
            batch_size=self.batch_list,
            shuffle=self.shuffle_list,
        )

    def get_runs(self):
        run = namedtuple('Run', self.params.keys())
        runs = []
        for val in product(*self.params.values()):
            runs.append(run(*val))

        return runs

    def hypersTune(self, runs):
        for run in runs:
            self.Tune(run)

    def Tune(self, run):
        train_set, train_loader = downloadData(batch_size=run.batch_size)
        comment = f'{run}'
        print(comment)
        global network

        network = Network(train_set, train_loader, in_level=1, conv_level_1=6, conv_level_2=12,
                          fc_level_1=120, fc_level_2=60, out_level=10, lr=0.01, batch_size=100,
                          shuffle=True)

        network.trainNetwork(epochs=self.epochs, comment=comment)


if __name__ == "__main__":

    torch.set_printoptions(linewidth=150)
    # torch.set_grad_enabled(True)  # reduce memory consumption
    train_set, train_loader = downloadData(batch_size=100)

    network = Network(train_set, train_loader,in_level=1, conv_level_1=6, conv_level_2=12,
                            fc_level_1=120, fc_level_2=60, out_level=10, batch_size=100,
                            lr=0.01, shuffle=True)  # create network

    if torch.cuda.is_available():
        network.to("cuda")

    network.trainNetwork(epochs=1)

    ##################################################################
    builder = RunBuilder(lr_list= [.01, .001], batch_list=[1000, 2000],
                         shuffle_list=[True, False], epochs=1)
    runs = builder.get_runs()
    builder.hypersTune(runs)
