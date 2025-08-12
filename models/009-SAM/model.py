import torch  # 导入 PyTorch 深度学习框架
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，方便我们定义自己的网络结构
from segment_anything import sam_model_registry  # 从 segment_anything 库中导入模型注册表，用于方便地创建 SAM 模型

class SAM(nn.Module):  # 定义一个名为 SAM 的类，它继承自 torch.nn.Module，这是所有 PyTorch 模型的基类
    def __init__(self, model_type='vit_b', checkpoint='sam_vit_b_01ec64.pth'):  # 类的初始化方法，用于创建类的实例时进行设置
        super(SAM, self).__init__()  # 调用父类 nn.Module 的初始化方法，这是必须的步骤
        # 这行代码是核心，它通过 sam_model_registry 来创建一个预训练的 SAM 模型实例
        # model_type 参数决定了使用哪个版本的 SAM 模型，例如 'vit_b' (Vision Transformer Base)
        # checkpoint 参数指定了预训练权重的路径，模型将加载这些权重
        self.model = sam_model_registry[model_type](checkpoint=checkpoint)

    def forward(self, images, prompts):  # 定义模型的前向传播逻辑，输入为图像和提示
        # SAM 的独特之处在于它是一个“可提示”的模型，需要图像和提示（prompts）作为输入
        # 提示可以是一个点、一个边界框、一个粗略的掩码或一段文本描述
        # 这里我们假设 prompts 是一个包含了具体提示信息的字典
        outputs = self.model(images, prompts)  # 将图像和提示传入预训练的 SAM 模型，获取输出
        return outputs  # 返回模型的输出，通常是一个包含分割掩码和质量分数的字典

    def get_image_encoder(self):  # 定义一个方法来获取 SAM 模型的图像编码器部分
        # 图像编码器负责将输入的图像转换为高维的特征向量
        return self.model.image_encoder  # 返回图像编码器模块

    def get_prompt_encoder(self):  # 定义一个方法来获取 SAM 模型的提示编码器部分
        # 提示编码器负责将各种类型的提示（点、框等）转换为特征向量
        return self.model.prompt_encoder  # 返回提示编码器模块

    def get_mask_decoder(self):  # 定义一个方法来获取 SAM 模型的掩码解码器部分
        # 掩码解码器负责融合图像特征和提示特征，并最终生成分割掩码
        return self.model.mask_decoder  # 返回掩码解码器模块