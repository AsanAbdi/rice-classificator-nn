import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Проверка устройства (GPU или CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Загрузка и предобработка данных
data_df = pd.read_csv("C:/Users/user/Desktop/STUDY/Rice Classification/rice-type-classification/riceClassification.csv")  # Укажите правильный путь
data_df.dropna(inplace=True)
data_df.drop(["id"], axis=1, inplace=True)

# Нормализация данных
for column in data_df.columns:
    data_df[column] = data_df[column] / data_df[column].abs().max()

# Разделение на признаки (X) и целевую переменную (Y)
X = np.array(data_df.iloc[:, :-1])
Y = np.array(data_df.iloc[:, -1])

# Разделение на тренировочный, валидационный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Класс Dataset для PyTorch
class RiceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.Y = torch.tensor(Y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

# Создание датасетов
training_data = RiceDataset(X_train, y_train)
validation_data = RiceDataset(X_val, y_val)
testing_data = RiceDataset(X_test, y_test)

# Создание DataLoader
train_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=8, shuffle=True)
testing_dataloader = DataLoader(testing_data, batch_size=8, shuffle=True)

# Определение нейросети
HIDDEN_NEURONS = 5

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.input_layer = nn.Linear(X.shape[1], HIDDEN_NEURONS)
        self.hidden_activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(HIDDEN_NEURONS, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_activation(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

# Инициализация модели
model = MyModel().to(device)
summary(model, (X.shape[1],))

# Функция потерь и оптимизатор
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=1e-3)

# Списки для хранения значений потерь и точности
total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []

# Обучение модели
epochs = 10
for epoch in range(epochs):
    total_acc_train = 0
    total_loss_train = 0
    total_acc_val = 0
    total_loss_val = 0

    # Тренировочный этап
    for data in train_dataloader:
        inputs, labels = data
        prediction = model(inputs).squeeze(1)
        batch_loss = criterion(prediction, labels)
        total_loss_train += batch_loss.item()
        acc = ((prediction.round() == labels).sum().item())
        total_acc_train += acc

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Валидация
    with torch.no_grad():
        for data in validation_dataloader:
            inputs, labels = data
            prediction = model(inputs).squeeze(1)
            batch_loss = criterion(prediction, labels)
            total_loss_val += batch_loss.item()
            acc = ((prediction.round() == labels).sum().item())
            total_acc_val += acc

    # Сохранение значений
    total_loss_train_plot.append(round(total_loss_train / len(train_dataloader), 4))
    total_loss_validation_plot.append(round(total_loss_val / len(validation_dataloader), 4))
    total_acc_train_plot.append(round(total_acc_train / training_data.__len__() * 100, 4))
    total_acc_validation_plot.append(round(total_acc_val / validation_data.__len__() * 100, 4))

    print(f"Epoch {epoch+1}/{epochs}:")
    print(f"Train Loss: {total_loss_train_plot[-1]}, Train Accuracy: {total_acc_train_plot[-1]}%")
    print(f"Validation Loss: {total_loss_validation_plot[-1]}, Validation Accuracy: {total_acc_validation_plot[-1]}%")
    print("=" * 30)

# Тестирование модели
with torch.no_grad():
    total_loss_test = 0
    total_accuracy_test = 0

    for data in testing_dataloader:
        inputs, labels = data
        prediction = model(inputs).squeeze(1)
        batch_loss_test = criterion(prediction, labels).item()
        total_loss_test += batch_loss_test
        acc = ((prediction.round() == labels).sum().item())
        total_accuracy_test += acc

    print(f"Test Accuracy: {round(total_accuracy_test / testing_data.__len__() * 100, 4)}%")

# Визуализация результатов
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# График потерь
axs[0].plot(total_loss_train_plot, label="Training Loss")
axs[0].plot(total_loss_validation_plot, label="Validation Loss")
axs[0].set_title("Training and Validation Loss over Epochs")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].legend()

# График точности
axs[1].plot(total_acc_train_plot, label="Training Accuracy")
axs[1].plot(total_acc_validation_plot, label="Validation Accuracy")
axs[1].set_title("Training and Validation Accuracy over Epochs")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Accuracy")
axs[1].legend()

plt.show()
