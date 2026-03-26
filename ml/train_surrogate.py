import torch
import numpy as np
from model import SurrogateModel


# load data
X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")

y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

# convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# model
model = SurrogateModel()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

loss_fn = torch.nn.MSELoss()

epochs = 40
batch_size = 512


for epoch in range(epochs):

    perm = torch.randperm(X_train.size(0))

    total_loss = 0

    for i in range(0, X_train.size(0), batch_size):

        idx = perm[i:i+batch_size]

        xb = X_train[idx]
        yb = y_train[idx]

        pred = model(xb)

        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"epoch {epoch+1} loss {total_loss:.4f}")


# evaluation
with torch.no_grad():

    pred = model(X_test)

    test_loss = loss_fn(pred, y_test)

print("\n test MSE:", test_loss.item())


# save model
torch.save(model.state_dict(), "models/surrogate.pt")

print("\nmodel saved → models/surrogate.pt")