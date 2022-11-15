import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikipediaNetwork, Actor, WebKB
from sklearn.model_selection import train_test_split
from torch_geometric.logging import init_wandb, log

from models import GCN
import models_multilayer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--wandb', action='store_true', help='Track experiment')

args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 7 homo: cora/cite-seer/pubmed/amazon computer, photos/coauthor CS, physics/
    # 6 heter: wiki chameleon/squirrel/actor co-occur/wisconsin/texas/cornell

    dataset_data = {
        'cora': Planetoid(root='/tmp/Cora', name='Cora'),
        'pubmed': Planetoid(root='/tmp/Cora', name='PubMed'),
        'citeseer': Planetoid(root='/tmp/Citeseer', name='Citeseer'),
        'cs': Coauthor(root='/tmp/CS', name='CS'),
        'physics': Coauthor(root='/tmp/physics', name='physics'),
        'computers': Amazon(root='/tmp/Computers', name='Computers'),
        'chameleon': WikipediaNetwork(root='/tmp/chameleon', name='chameleon'),
        'squirrel': WikipediaNetwork(root='/tmp/squirrel', name='squirrel'),
        'actor': Actor(root='/tmp/actor'),
        'Cornell': WebKB(root='/tmp/Cornell', name='Cornell'),
        'Texas': WebKB(root='/tmp/Texas', name='Texas'),
        'Wisconsin': WebKB(root='/tmp/Texas', name='Wisconsin')
    }

    gnn_classes = [cls for cls in map(models_multilayer.__dict__.get, models_multilayer.__all__)]

    for g in gnn_classes:
        print("==============")
        log(model=g)

        for data_name, data_content in dataset_data.items():

            for layer_num in range(1, 8, 2):
                path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', data_name)
                dataset = data_content.shuffle()
                data = dataset[0]

                # train / test split
                data_num = data.num_nodes
                train_idx, test_idx = train_test_split(list(range(data_num)), test_size=0.20, random_state=42)
                train_idx, val_idx = train_test_split(train_idx, test_size=0.12, random_state=1)  # 0.25 x 0.8 = 0.2

                data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                data.train_mask[train_idx] = 1
                data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                data.test_mask[test_idx] = 1
                data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                data.val_mask[val_idx] = 1

                # setup model
                model = g(dataset.num_features, args.hidden_channels, dataset.num_classes, layer_num)
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
                try:
                    for epoch in range(1, args.epochs + 1):

                        loss = train()
                        train_acc, val_acc, tmp_test_acc = test()
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            final_test_acc = tmp_test_acc
                        # if epoch % 10 == 0:
                        #     log(Data=data_name, Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)

                except Exception as e:
                    print('error')
                log(Data=data_name, Layer_N=layer_num, Test=final_test_acc)
