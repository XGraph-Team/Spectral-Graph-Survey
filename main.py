from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikipediaNetwork, Actor, WebKB

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':

    # the metric of homo/heter phily?
    # run the experiment and see how to explain

    # 7 homo: cora/cite-seer/pubmed/amazon computer, photos/coauthor CS, physics/
    #
    # dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    # dataset = Coauthor(root='/tmp/CS', name='CS')
    # dataset = Coauthor(root='/tmp/physics', name='physics')
    # dataset = Amazon(root='/tmp/Computers', name='Computers')

    # 6 heter: wiki chameleon/squirrel/actor co-occur/wisconsin/texas/cornell
    #
    # dataset = WikipediaNetwork(root='/tmp/chameleon', name='chameleon')
    # dataset = WikipediaNetwork(root='/tmp/squirrel', name='squirrel')
    # dataset = Actor(root='/tmp/actor')
    # dataset = WebKB(root='/tmp/Cornell', name='Cornell')
    # dataset = WebKB(root='/tmp/Texas', name='Texas')
    # dataset = WebKB(root='/tmp/Texas', name='Wisconsin')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        # not every dataset has train mast, implement yourself
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')