import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义基本的残差块 (BasicBlock)
class BasicBlock(nn.Module):
    expansion = 1  # 表明输出通道数相对于输入通道数的扩展比例

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # 批量归一化层1
        self.bn1 = nn.BatchNorm2d(out_channels)
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 批量归一化层2
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 快捷连接 (Shortcut connection)
        self.shortcut = nn.Sequential()
        # 如果输入和输出的维度不匹配（例如stride不为1，或通道数不同），则需要通过1x1卷积调整维度
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        # 主路径
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 主路径输出与快捷连接输出相加
        out += self.shortcut(x)
        # 应用ReLU激活函数
        out = self.relu(out)
        return out

# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 四个残差层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 全局平均池化层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层 (分类器)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    # 创建一个残差层（由多个残差块组成）
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 定义ResNet18模型
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# --- 模型训练代码 ---

# 1. 设置设备 (CPU or GPU)
def train_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} for training.')

    # 2. 数据预处理和加载
    # 定义数据增强和归一化
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # 随机裁剪
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.ToTensor(), # 转换为Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # 归一化
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载CIFAR-10数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # 3. 初始化模型、损失函数和优化器
    # 实例化ResNet18模型并移动到指定设备
    net = ResNet18().to(device)

    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义SGD优化器
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # 4. 训练和验证循环
    def train(epoch):
        print(f'\nEpoch: {epoch}')
        net.train() # 设置为训练模式
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device) # 将数据移动到设备
            optimizer.zero_grad() # 清空梯度
            outputs = net(inputs) # 前向传播
            loss = criterion(outputs, targets) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新权重

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(trainloader)} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')

    def test(epoch):
        net.eval() # 设置为评估模式
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad(): # 禁用梯度计算
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # 打印测试结果
        print(f'Test Results | Loss: {test_loss/len(testloader):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')

    # 开始训练
    for epoch in range(200):
        train(epoch)
        test(epoch)
        scheduler.step()

# 如果作为主程序运行，则开始训练
if __name__ == '__main__':
    train_model()