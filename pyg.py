import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.nn import SplineConv, GCNConv
from torch_geometric.datasets import TUDataset, Planetoid, Coauthor, Amazon
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import MLP, GINConv, global_add_pool

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()





if __name__ == '__main__':




    # 7 homo: cora/cite-seer/pubmed/amazon computer, photos/coauthor CS, physics/
    dataset_data = {
        # 'cora': Planetoid(root='/tmp/Cora', name='Cora'),
        # 'pubmed': Planetoid(root='/tmp/Cora', name='PubMed'),
        # 'citeseer': Planetoid(root='/tmp/Citeseer', name='Citeseer'),
        'cs': Coauthor(root='/tmp/CS', name='CS'),
        # 'physics': Coauthor(root='/tmp/physics', name='physics'),
        # 'computers': Amazon(root='/tmp/Computers', name='Computers')
    }

    for data_name, data_content in dataset_data.items():

        path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', data_name)
        dataset = data_content.shuffle()
        data = dataset[0]

        # train / test split
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[:data.num_nodes - 1000] = 1

        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[data.num_nodes - 500:] = 1

        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[data.num_nodes - 1000:data.num_nodes - 500] = 1


        class Net(torch.nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels):
                super().__init__()
                self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
                self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)

            def forward(self, x, edge_index, edge_weight=None):
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv1(x, edge_index, edge_weight).relu()
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, edge_index, edge_weight)
                return x


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net(dataset.num_features, args.hidden_channels, dataset.num_classes)
        model, data = model.to(device), data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-3)


        def train():
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_weight)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            return float(loss)


        def test():
            model.eval()
            pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=-1)

            accs = []
            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
            return accs


        def validate():
            model.eval()
            logits, accs = model(), []
            for _, mask in data('val_mask'):
                pred = logits[mask].max(1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                accs.append(acc)
            return accs


        best_val_acc = final_test_acc = 0
        for epoch in range(1, args.epochs + 1):
            loss = train()
            train_acc, val_acc, tmp_test_acc = test()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
