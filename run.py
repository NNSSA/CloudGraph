import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch_geometric
from typing import List
from torch import Tensor
from torch_cluster import radius_graph, knn_graph
from torch_geometric.nn import GCNConv, MessagePassing
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

point_data = np.load("data_halos.npy")
print(
    point_data.shape
)  # (n_data, n_points, n_features); features are (x, y, z, vx, vy, vz, mass)
theta_train = pd.read_csv("cosmology.csv")
sigma8_labels = np.array(theta_train.loc[:, "sigma_8"])

Ntot = point_data.shape[0]
Ntrain = 1700
Ntest = Ntot - Ntrain
Nbatch = 10
knn = 20

train_data = torch.tensor(
    point_data[:Ntrain, :, :3], dtype=torch.float32, device=device
)
train_labels = torch.tensor(sigma8_labels[:Ntrain], dtype=torch.float32, device=device)
test_data = torch.tensor(
    point_data[Ntrain : Ntrain + Ntest, :, :3], dtype=torch.float32, device=device
)
test_labels = torch.tensor(
    sigma8_labels[Ntrain : Ntrain + Ntest], dtype=torch.float32, device=device
)

train_batch_indices = torch.repeat_interleave(torch.arange(Nbatch), train_data.shape[1])
test_batch_indices = torch.repeat_interleave(torch.arange(Ntest), test_data.shape[1])

# train_batched = torch.repeat_interleave(torch.arange(train_data.shape[0]), train_data.shape[1])
# test_batched = torch.repeat_interleave(torch.arange(test_data.shape[0]), test_data.shape[1])
# train_edge_index = knn_graph(train_data.view(-1, 3), k=knn, batch=train_batched)
# test_edge_index = knn_graph(test_data.view(-1, 3), k=knn, batch=test_batched)

train_loader = torch.utils.data.DataLoader(
    [[train_data[i], train_labels[i]] for i in range(Ntrain)],
    batch_size=Nbatch,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    [[test_data[i], test_labels[i]] for i in range(Ntest)],
    batch_size=Ntest,
    shuffle=True,
)

###############################################################################
###############################################################################


def get_mlp(in_channels=3, out_channels=64, hidden_layers=[64, 64]):
    layers = [torch.nn.Linear(in_channels, hidden_layers[0]), torch.nn.ReLU()]
    for i in range(len(hidden_layers) - 1):
        layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(hidden_layers[-1], out_channels))
    return torch.nn.Sequential(*layers)


# Define EdgeUpdate class for updating edge attributes
class EdgeUpdate(torch.nn.Module):
    def __init__(
        self, edge_in_channels: int, edge_out_channels: int, hidden_layers: List[int]
    ):
        super().__init__()
        self.mlp = get_mlp(
            in_channels=edge_in_channels,
            out_channels=edge_out_channels,
            hidden_layers=hidden_layers,
        )

    def forward(self, h_i: Tensor, h_j: Tensor, edge_attr: Tensor, u: Tensor) -> Tensor:
        inputs_to_concat = [h_i, h_j]
        if edge_attr is not None:
            inputs_to_concat.append(edge_attr)
        if u is not None:
            inputs_to_concat.append(u)
        inputs = torch.concat(inputs_to_concat, dim=-1)
        return self.mlp(inputs)


# Define NodeUpdate class for updating node features
class NodeUpdate(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_layers: List[int],
        aggr: str = "add",
    ):
        super().__init__(aggr=aggr)
        self.mlp = get_mlp(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_layers=hidden_layers,
        )

    def forward(
        self, h: Tensor, edge_index: Tensor, edge_attr: Tensor, u: Tensor
    ) -> Tensor:
        msg = self.propagate(edge_index, edge_attr=edge_attr)
        to_concat = [h, msg]
        if u is not None:
            to_concat.append(u)
        input = torch.concat(to_concat, dim=-1)
        return self.mlp(input)

    def message(self, edge_attr: Tensor) -> Tensor:
        return edge_attr


# Define the GraphLayer class
class GraphLayer(torch.nn.Module):
    def __init__(
        self,
        node_in_channels: int = 2,
        node_out_channels: int = 1,
        edge_in_channels: int = 2,
        hidden_layers: List[int] = [128, 128, 128],
        edge_out_channels: int = 16,
        global_in_channels: int = 0,
    ):
        super().__init__()
        self.edge_update = EdgeUpdate(
            edge_in_channels=node_in_channels * 2
            + edge_in_channels
            + global_in_channels,
            edge_out_channels=edge_out_channels,
            hidden_layers=hidden_layers,
        )
        self.node_update = NodeUpdate(
            in_channels=node_in_channels + edge_out_channels,
            out_channels=node_out_channels,
            hidden_layers=hidden_layers,
        )

    def forward(self, h, edge_index, edge_attr, u=None):
        row, col = edge_index
        edge_attr = self.edge_update(h[row], h[col], edge_attr, u)
        h_out = self.node_update(h, edge_index, edge_attr, u)
        return h_out, edge_attr


# Define the GNN Multilayer class
class GNN(torch.nn.Module):
    def __init__(
        self,
        node_in_channels: int = 3,
        node_embedding_channels: int = 1,
        node_out_channels: int = 1,
        edge_in_channels: int = 3,
        edge_embedding_channels: int = 16,
        edge_out_channels: int = 16,
        hidden_layers: List[int] = [128, 128, 128],
        num_layers: int = 2,
    ):
        super().__init__()
        self.graph_layers = torch.nn.ModuleList(
            [
                GraphLayer(
                    node_in_channels=node_in_channels
                    if i == 0
                    else node_embedding_channels,
                    node_out_channels=node_embedding_channels
                    if i < num_layers - 1
                    else node_out_channels,
                    edge_in_channels=edge_in_channels
                    if i == 0
                    else edge_embedding_channels,
                    edge_out_channels=edge_embedding_channels
                    if i < num_layers - 1
                    else edge_out_channels,
                    global_in_channels=0,
                    hidden_layers=hidden_layers,
                )
                for i in range(num_layers)
            ]
        )
        self.readout_layer = torch.nn.Linear(node_out_channels, 1)

    def forward(self, h: Tensor, batch: Tensor) -> Tensor:
        """
        h: [Nbatch x Npoints_in_cloud, 3]
        edge_index: [2, knn x Nbatch x Npoints_in_cloud]
        edge_attr: [Nbatch x Npoints_in_cloud x Npoints_in_cloud]
        """

        edge_index = knn_graph(h, k=knn, batch=batch)

        # Compute distances as edge features
        row, col = edge_index
        edge_attr = torch.norm(h[row] - h[col], dim=1).view(-1, 1)

        for i, graph_layer in enumerate(self.graph_layers):
            h, edge_attr = graph_layer(h=h, edge_index=edge_index, edge_attr=edge_attr)

        h = torch_geometric.nn.global_mean_pool(h, batch)
        h = self.readout_layer(h)
        return h


gnn = GNN(
    node_in_channels=3,
    node_embedding_channels=1,
    node_out_channels=1,
    edge_in_channels=1,
    edge_embedding_channels=16,
    edge_out_channels=16,
    hidden_layers=[128, 128],
    num_layers=2,
).to(device)

train_loss_list = []
test_loss_list = []
lr = 1e-3
epochs = 3
optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)
MSE_loss = torch.nn.MSELoss()

###############################################################################
###############################################################################

for epoch in range(epochs):
    gnn.train(True)
    train_running_loss = 0.0
    with tqdm(
        total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"
    ) as pbar:
        for train_data_batch, train_labels_batch in train_loader:
            train_data_batch = train_data_batch.to(device)
            train_labels_batch = train_labels_batch.to(device)

            model_pred_train = gnn(train_data_batch.view(-1, 3), train_batch_indices)
            train_loss = MSE_loss(model_pred_train, train_labels_batch.view(-1, 1))
            train_running_loss += train_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            pbar.set_postfix(
                {"Train loss": f"{(train_running_loss / int(Ntrain/Nbatch)):.4f}"}
            )
            pbar.update()

    gnn.train(False)
    test_running_loss = 0
    with tqdm(
        total=len(test_loader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"
    ) as qbar:
        with torch.no_grad():
            for test_data_batch, test_labels_batch in test_loader:
                test_data_batch = test_data_batch.to(device)
                test_labels_batch = test_labels_batch.to(device)

                model_pred_test = gnn(test_data_batch.view(-1, 3), test_batch_indices)
                test_loss = MSE_loss(model_pred_test, test_labels_batch.view(-1, 1))
                test_running_loss += test_loss.item()

                qbar.set_postfix(
                    {"Test loss": f"{(test_running_loss / int(Ntest/Ntest)):.4f}"}
                )
                qbar.update()

    print("Train loss {}".format(train_running_loss / int(Ntrain / Nbatch)))
    print("Test loss {}".format(test_running_loss / int(Ntest / Nbatch)))
    train_loss_list.append(train_running_loss / int(Ntrain / Nbatch))
    test_loss_list.append(test_running_loss / int(Ntest / Ntest))

    if (epoch + 1) % 2 == 0:
        optimizer.param_groups[0]["lr"] /= 2.0
        print(optimizer.param_groups[0]["lr"])

###############################################################################
###############################################################################

plt.figure()
plt.semilogy(train_loss_list, color="blue", label="train loss")
plt.semilogy(test_loss_list, color="red", label="test loss")
plt.legend()
plt.savefig("loss.png")

rands = np.random.uniform(0, Ntrain, 20)
train_batched = torch.repeat_interleave(
    torch.arange(train_data[rands].shape[0]), train_data[rands].shape[1]
)
model_pred_train = (
    gnn(train_data[rands].view(-1, 3), train_batched).cpu().detach().numpy()
)
actual_values_train = train_labels[rands].cpu().detach().numpy()

for numtrain in range(len(rands)):
    model_value = model_pred_train[numtrain][0]
    actual_value = actual_values_train[numtrain]
    print("Train comparison: ", model_value, actual_value)

test_batched = torch.repeat_interleave(
    torch.arange(test_data.shape[0]), test_data.shape[1]
)
model_pred_test = gnn(test_data.view(-1, 3), test_batched)

accuracy = 0.0
for numtest in range(Ntest):
    model_value = model_pred_test[numtest].cpu().detach().numpy()[0]
    actual_value = test_labels[numtest].cpu().detach().numpy()
    print("Test comparison: ", model_value, actual_value)
    if np.abs(model_value - actual_value) / actual_value < 0.1:
        accuracy += 1.0

print("Accuracy: ", accuracy / Ntest)

plt.figure()
plt.scatter(
    model_pred_train[:, 0],
    actual_values_train,
    s=10,
    alpha=0.6,
    color="red",
    label="train",
)
plt.scatter(
    model_pred_test.cpu().detach().numpy()[:, 0],
    test_labels.cpu().detach().numpy(),
    s=10,
    alpha=0.6,
    color="blue",
    label="test",
)
plt.legend()
plt.savefig("results.png")
