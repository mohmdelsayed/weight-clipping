import torch, torchvision
from core.network.resent18 import ResNet18

class CIFAR10Nonstationary:
    def __init__(self, name="cifar10_nonstationary", batch_size=125, is_non_stationary=True, n_epochs=10, n_iteration_per_epoch=400):
        self.train_dataset1 = self.get_dataset(train=True)
        self.test_dataset = self.get_dataset(train=False)
        self.batch_size = batch_size
        self.iterator = self.generator1()
        self.n_inputs = 3 * 32 * 32
        self.n_outputs = 10
        self.name = name
        self.time = 0
        self.n_samples = n_epochs * n_iteration_per_epoch
        if is_non_stationary:
            self.n_iterations = n_epochs * n_iteration_per_epoch // 2
            self.n_iteration_per_epoch = n_iteration_per_epoch // 2
        else:
            self.n_iterations = n_epochs * n_iteration_per_epoch
            self.n_iteration_per_epoch = n_iteration_per_epoch
        
        self.criterion = "cross_entropy"
        self.is_non_stationary = is_non_stationary
        if self.is_non_stationary:
            self.train_dataset1, self.train_dataset2 = torch.utils.data.random_split(self.train_dataset1, [25000, 25000])
            self.iterator = self.generator1()

    def __next__(self):
        if self.is_non_stationary and self.time >= self.n_iterations:
            try:
                sample = next(self.iterator)
                self.time += 1
                return sample
            except:
                self.iterator = self.generator2()
                sample = next(self.iterator)
                self.time += 1
                return sample
        else:
            try:
                sample = next(self.iterator)
                self.time += 1
                return sample
            except:
                self.iterator = self.generator1()
                sample = next(self.iterator)
                self.time += 1
                return sample

    def __str__(self) -> str:
        return self.name

    def __iter__(self):
        return self

    def generator1(self):
        return iter(self.get_dataloader(self.train_dataset1, self.batch_size))

    def generator2(self):
        return iter(self.get_dataloader(self.train_dataset2, self.batch_size))

    def get_test_dataloader(self):
        return iter(self.get_dataloader(self.test_dataset, batch_size=self.batch_size))

    def get_dataset(self, train=True):
        return torchvision.datasets.CIFAR10(
            "dataset",
            train=train,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )

    def get_dataloader(self, dataset, batch_size=125, shuffle=True):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    network = ResNet18()
    n_iteration_per_epoch = 400
    n_epochs = 700
    task = CIFAR10Nonstationary(is_non_stationary=True, n_epochs=n_epochs, n_iteration_per_epoch=n_iteration_per_epoch)

    test_loader = task.get_test_dataloader()
    for i, (x, y) in enumerate(test_loader):
        output = network(x)
        print(torch.argmax(output, dim=-1).shape, i)