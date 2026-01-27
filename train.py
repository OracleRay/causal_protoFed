import torch
from torch.utils.data import DataLoader
from tqdm import *

from federate.server import FederatedServer as Server
from federate.client import FederatedClient as Client
from model.st_model import LocalTrafficModel as Model
from utils.data_utils import generate_dataset
from utils.metrics import metric_func
import os


os.environ['LOKY_MAX_CPU_COUNT'] = '16'  # 设置CPU核心数

@torch.no_grad()
def evaluate_model(target_means, target_stds, model, dataloader, device):
    model.eval()
    all_pred, all_y = [], []
    for x, y, _ in dataloader:
        x = x.to(device).float()
        y = y.to(device).float()
        pred, _, _, _ = model(x)

        # 反标准化用于评估
        y = y * target_stds + target_means
        pred = pred * target_stds + target_means

        all_pred.append(pred.detach().cpu())
        all_y.append(y.detach().cpu())
    pred = torch.cat(all_pred, dim=0)
    y = torch.cat(all_y, dim=0)
    return metric_func(pred, y)


def main_federated():
    # 任务设定：pems-bay 为服务器城市，其余为客户端城市；训练四城都参与，评估只用三客户端城市
    server_city = "pems-bay"
    client_cities = ["chengdu", "metr-la", "shenzhen"]
    all_cities = [server_city] + client_cities

    # 数据切片参数
    options = {
        "his_num": 12,
        "pred_num": 6,
        "seed": 0,
        "target_city": server_city,
        "feature_idx": 0,  # dataset.npy 的第 0 个特征作为预测目标
    }

    # 训练参数
    hidden_dim = 32
    num_env_classes = 3
    rounds = 10
    local_epochs = 1
    batch_size = 32

    # input_dim = his_num（单一特征）；output_dim = pred_num（多步预测）
    config = {
        "input_dim": options["his_num"],
        "hidden_dim": hidden_dim,
        "output_dim": options["pred_num"],
        "num_env_classes": num_env_classes,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建全局模型 & 服务器
    global_model = Model(**config).to(device)
    server = Server(global_model)

    # 为每个城市构建 client + dataloader
    clients = {}
    train_loaders, test_loaders = {}, {}
    target_means, target_stds = None, None
    for idx, city in enumerate(all_cities):
        train_ds, test_ds = generate_dataset(options, city)
        if city == server_city:  # 记录目标城市的均值，方便后续反标准化
            target_means = test_ds.dataset_means[0]
            target_stds = test_ds.dataset_stds[0]
        train_loaders[city] = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loaders[city] = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        clients[city] = Client(idx, config)

    global_weights = global_model.state_dict()
    global_protos = None

    print("开始联邦训练...")
    print(f"服务器城市: {server_city}；客户端城市: {client_cities}")

    client_weights, client_protos, client_loss = {}, {}, {}
    for r in range(rounds):
        print(f"--- Round {r + 1} / {rounds} ---")

        # 所有城市（包含服务器城市）都作为“客户端”参与本地训练
        for city in all_cities:
            client = clients[city]

            # 更新本地参数
            if r > 0:
                new_client_weights = client.update_weights(global_weights, client_weights[city])
                client.model.load_state_dict(new_client_weights)

            # 本地多 epoch
            weights, protos, loss = None, None, None
            for _ in tqdm(range(local_epochs)):
                weights, protos, loss = client.train_epoch(train_loaders[city], global_protos)

            client_weights[city] = weights
            client_protos[city] = protos
            client_loss[city] = loss

        # 服务器聚合（使用所有参与方的更新）
        global_weights, global_protos = server.aggregate(client_weights, client_protos, client_loss)
        global_model.load_state_dict(global_weights)

        # 评估目标城市的预测准确度
        metrics = evaluate_model(target_means, target_stds, global_model, test_loaders[server_city], device)
        print(f"{server_city}: RMSE: {metrics['RMSE']} | MAE: {metrics['MAE']}")


if __name__ == "__main__":
    main_federated()