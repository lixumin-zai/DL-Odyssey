import torch  # 导入 PyTorch 深度学习框架
import pytorch_lightning as pl  # 导入 PyTorch Lightning 框架，用于简化训练流程
from torch.utils.data import DataLoader  # 从 PyTorch 中导入 DataLoader，用于高效地加载数据
from segment_anything import sam_model_registry  # 从 segment_anything 库中导入模型注册表
from dataset import SAMDataset  # 从我们自己编写的 dataset.py 文件中导入 SAMDataset 类
import torch.optim as optim  # 导入 PyTorch 的优化器模块，例如 Adam

class SAMLightning(pl.LightningModule):  # 定义一个名为 SAMLightning 的类，它继承自 pytorch_lightning.LightningModule
    def __init__(self, model_type='vit_b', checkpoint='sam_vit_b_01ec64.pth', learning_rate=1e-4):  # 类的初始化方法
        super().__init__()  # 调用父类的初始化方法
        # self.save_hyperparameters() 会自动将传入的参数（如 model_type, learning_rate）保存到 self.hparams 中，方便后续访问和保存
        self.save_hyperparameters()
        # 使用模型注册表和指定的模型类型、权重路径来创建 SAM 模型实例
        self.model = sam_model_registry[self.hparams.model_type](checkpoint=self.hparams.checkpoint)
        # 定义损失函数。这里使用均方误差损失（MSELoss），用于比较模型输出的掩码和真实的掩码
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, images, prompts):  # 定义模型的前向传播逻辑
        # 直接调用内部的 SAM 模型的 forward 方法
        return self.model(images, prompts)

    def training_step(self, batch, batch_idx):  # 定义单个训练步骤
        # 从数据加载器中获取一个批次的数据，包含图像、真实掩码和图像路径
        images, masks, _ = batch
        # 创建一个空的提示（prompt）。在实际应用中，你需要根据任务需求来构建具体的提示，例如提供点坐标或边界框
        # 这里为了简化，我们不提供任何具体的空间提示，让模型尝试在没有引导的情况下分割
        prompts = {
            "point_coords": None,  # 点提示的坐标
            "point_labels": None,  # 点提示的标签（前景点/背景点）
            "mask_input": None,    # 输入的粗略掩码提示
            "box": None,           # 边界框提示
            "multimask_output": False,  # 是否输出多个可能的掩码
        }
        # 执行前向传播，获取模型的输出
        outputs = self(images, prompts)
        # 计算模型预测的掩码和真实掩码之间的损失
        loss = self.loss_fn(outputs['masks'], masks)
        # 使用 self.log() 方法记录训练损失，这对于在 TensorBoard 等工具中监控训练过程非常有用
        self.log('train_loss', loss)
        return loss  # 返回计算出的损失，PyTorch Lightning 会自动处理反向传播和优化器步骤

    def validation_step(self, batch, batch_idx):  # 定义单个验证步骤，逻辑与 training_step 非常相似
        images, masks, _ = batch
        prompts = {
            "point_coords": None,
            "point_labels": None,
            "mask_input": None,
            "box": None,
            "multimask_output": False,
        }
        outputs = self(images, prompts)
        loss = self.loss_fn(outputs['masks'], masks)
        # 记录验证损失，用于评估模型在未见过的数据上的性能
        self.log('val_loss', loss)

    def configure_optimizers(self):  # 定义如何配置优化器
        # 创建一个 Adam 优化器，传入模型的参数和在初始化时定义的学习率
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer  # 返回优化器

if __name__ == '__main__':  # 这是一个标准的 Python 写法，确保以下代码只在直接运行此脚本时执行
    # --- 数据集路径配置 ---
    # !!! 重要提示: 请将这里的路径替换为您自己存放图像和掩码的实际路径 !!!
    IMAGE_DIR = 'path/to/your/images'
    MASK_DIR = 'path/to/your/masks'
    
    # --- 数据集和数据加载器创建 ---
    # 创建训练数据集实例
    train_dataset = SAMDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR)
    # 创建验证数据集实例（这里为了简单，使用了和训练集相同的路径，实际应用中应该使用独立的验证集）
    val_dataset = SAMDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR)
    # 创建训练数据加载器，设置批次大小为2，并开启 shuffle 随机打乱数据
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    # 创建验证数据加载器
    val_loader = DataLoader(val_dataset, batch_size=2)
    
    # --- 模型和训练器创建 ---
    # 创建 SAMLightning 模型实例
    model = SAMLightning()
    # 创建 PyTorch Lightning 的训练器（Trainer）
    # max_epochs=10 表示最多训练10个周期
    # gpus=1 表示使用1个GPU进行训练。如果没有GPU，可以设置为 gpus=0 或完全移除该参数
    trainer = pl.Trainer(max_epochs=10, gpus=1)
    
    # --- 开始训练 ---
    # 调用 trainer.fit() 方法开始训练过程
    # 需要传入模型、训练数据加载器和验证数据加载器
    trainer.fit(model, train_loader, val_loader)