#### Running Unnormalized GCN with karate club dataset
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt


def gcn_msg(edges):
    # just pass source node's embedding to destination node
    return {'m': edges.src['h']}


def gcn_reduce(nodes):
    # sum the embedding of all neighbor nodes
    return {'h': torch.sum(nodes.mailbox['m'], dim=1)}


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        # define a fully connected layer to store W
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # perform one-pass of updates on graph
        # return the updated embeddings of all nodes
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


class Net(nn.Module):
    # Two layer GCN for prediction on 34-feature network
    # prediction two classes
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(34, 16)
        self.layer2 = GCNLayer(16, 2)

    def forward(self, g, features):
        # input features are being used to learn some vector per node
        x = F.relu(self.layer1(g, features))
        # learnt vector is refined by non-linear activation and used
        # to learn next a vector on next layer
        x = self.layer2(g, x)
        return x


def evaluate(model, g, features, labels, mask):
    model.eval()
    # disable gradient computation
    with torch.no_grad():
        # compute embeddings
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        # predict class 1 for node x if logits[x][1] > logits[x][0]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        # accuracy computation
        return correct.item() * 1.0 / len(labels)


def load_karate_club():
    # generate training and testing masks
    # alongisde loading dataset
    g = nx.karate_club_graph()
    labels = []
    for i in g.nodes():
        n = g.node.data()[i]
        if n['club'] == 'Officer':
            labels.append(1)
        else:
            labels.append(0)

    # one-hot encoded node id
    feats = np.eye(len(g.nodes()))
    train_mask = np.zeros(len(g.nodes))
    # only first and last node for training
    train_mask[[0, train_mask.shape[0]-1]] = 1
    # all nodes for testing
    test_mask = np.ones(len(g.nodes))

    # convert everything to pytorch variables
    g = dgl.DGLGraph(g)
    feats = torch.FloatTensor(feats)
    train_mask = torch.BoolTensor(train_mask)
    test_mask = torch.BoolTensor(test_mask)
    labels = torch.LongTensor(labels)
    return g, feats, labels, train_mask, test_mask


net = Net()
print(net)

g, features, labels, train_mask, test_mask = load_karate_club()
# simple Adam optimizer. LR = 1e-2 because features are already in a small
# range of 0 to 1
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
dur = []
selected_epochs = set(range(0, 50, 5))
to_visualize = []
for epoch in range(26):
    if epoch >= 3:
        t0 = time.time()

    net.train()
    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    acc = evaluate(net, g, features, labels, test_mask)

    if epoch in selected_epochs:
        to_visualize.append((epoch, logits.detach().numpy(), acc))
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
        epoch, loss.item(), acc, np.mean(dur)))

filter_0 = np.where(np.array(labels) == 0)
filter_1 = np.where(np.array(labels) == 1)
fig, axs = plt.subplots(2, len(to_visualize)//2)
for ax, (epoch, element, acc) in zip(axs.flatten(), to_visualize):
    ax.set_title("Epoch {}. Accuracy {:0.2f}".format(epoch, acc))
    ax.scatter(element[filter_0][:, 0], element[filter_0][:, 1], label='0')
    ax.scatter(element[filter_1][:, 0], element[filter_1][:, 1], label='1')
    ax.legend()
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label='Class Boundary')
    ax.legend()

fig.text(0.5, 0.01, 'X1', ha='center')
fig.text(0.01, 0.5, 'X2', va='center', rotation='vertical')
plt.show()
