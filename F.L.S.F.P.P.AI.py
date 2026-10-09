import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# =====================================
# 1. Device
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)


# =====================================
# 2. Load MNIST
# =====================================

dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)


# =====================================
# 3. Split data between clients
# =====================================

num_clients = 5

client_data = torch.utils.data.random_split(
    dataset,
    [len(dataset) // num_clients]
    * num_clients
)


# =====================================
# 4. Neural network
# =====================================

class SimpleNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                28 * 28,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self, x):

        return self.network(x)


# =====================================
# 5. Train one client
# =====================================

def train_client(
    global_model,
    client_dataset,
    epochs=1
):

    model = copy.deepcopy(
        global_model
    )

    model.to(device)

    loader = DataLoader(
        client_dataset,
        batch_size=64,
        shuffle=True
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(images)

            loss = criterion(
                output,
                labels
            )

            loss.backward()

            optimizer.step()

    return model.state_dict()


# =====================================
# 6. FedAvg
# =====================================

def federated_average(
    client_models
):

    global_model = copy.deepcopy(
        client_models[0]
    )

    for key in global_model:

        global_model[key] = torch.stack(
            [
                client[key].float()
                for client in client_models
            ],
            dim=0
        ).mean(dim=0)

    return global_model


# =====================================
# 7. Evaluation
# =====================================

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


def evaluate(model):

    loader = DataLoader(
        test_dataset,
        batch_size=128
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total


# =====================================
# 8. Start global model
# =====================================

global_model = SimpleNN()

global_model.to(device)


# =====================================
# 9. Federated training
# =====================================

rounds = 10

for round_number in range(rounds):

    client_models = []

    print(
        f"\nFederated Round "
        f"{round_number + 1}"
    )

    # Train each client
    for client_id in range(
        num_clients
    ):

        local_model = train_client(
            global_model,
            client_data[client_id],
            epochs=1
        )

        client_models.append(
            local_model
        )

        print(
            f"Client {client_id + 1} finished"
        )


    # Combine model updates
    new_global_state = federated_average(
        client_models
    )


    global_model.load_state_dict(
        new_global_state
    )


    accuracy = evaluate(
        global_model
    )

    print(
        "Global Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )