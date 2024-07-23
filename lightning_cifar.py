import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
from models import ResNet18
from utils import progress_bar

class CIFAR10Lightning(pl.LightningModule):
    def __init__(self, lr=0.1):
        super(CIFAR10Lightning, self).__init__()
        self.lr = lr
        self.best_acc = 0
        self.start_epoch = 0

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.transform_train)
        self.trainloader = DataLoader(
            self.trainset, batch_size=128, shuffle=True, num_workers=8)

        self.testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.transform_test)
        self.testloader = DataLoader(
            self.testset, batch_size=20, shuffle=False, num_workers=2)

        self.net = ResNet18()

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr,
                                   momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        self.val_loss = 0
        self.val_correct = 0
        self.val_total = 0
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        return loss

    def on_training_epoch_end(self, outputs):
        self.avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        #return {'loss': avg_loss}

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        total = targets.size(0)
        self.val_loss += loss
        self.val_correct += correct
        self.val_total += total
        #self.log("presision",self.val_correct/self.val_total,prog_bar=True)
        #return {'loss': loss, 'correct': correct, 'total': total}
    
    def on_validation_epoch_end(self):
        
        total_correct = self.val_correct
        total_samples = self.val_total
        avg_loss = self.val_loss /total_samples
        acc = total_correct / total_samples * 100
        self.val_loss = 0
        self.val_total = 0
        self.val_correct = 0
        if acc > self.best_acc:
            self.best_acc = acc
            checkpoint = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': self.current_epoch,
            }
            torch.save(checkpoint, './checkpoint/ckpt.pth')
        self.log("acc",acc,prog_bar=True)
        #return {'avg_test_loss': avg_loss, 'test_accuracy': acc}
        #print('avg_test_loss', avg_loss, 'test_accuracy', acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    model = CIFAR10Lightning(lr=args.lr)

    if args.resume:
        # Load checkpoint.
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        model.load_state_dict(checkpoint['net'])
        model.best_acc = checkpoint['acc']
        model.start_epoch = checkpoint['epoch']

    trainer = pl.Trainer(max_epochs=200, accelerator='gpu')
    trainer.fit(model,train_dataloaders=model.trainloader,val_dataloaders=model.testloader)
    trainer.test()
