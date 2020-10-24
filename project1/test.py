import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.cluster import KMeans
from sklearn import metrics

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ])),
  batch_size=10000, shuffle=False)
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
X = example_data.view(-1,784)
Y = example_targets


kmeans = KMeans(n_clusters=10, n_init=20, n_jobs=4)
# Train K-Means.
y_pred_kmeans = kmeans.fit_predict(X)
# Evaluate the K-Means clustering accuracy.
acc = metrics.accuracy_score(Y, y_pred_kmeans)

print(acc)