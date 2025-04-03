import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import Dataset, DataLoader

"""
ImageQualityEvaluator: 图像质量评估工具

1. FID（Fréchet Inception Distance）：衡量生成图与真实图在特征空间中的分布差异，越小越好。
2. IS（Inception Score）：衡量生成图是否语义清晰、类别多样，越大越好。
3. CLIP Score：衡量生成图与文本描述之间的语义相似度，越大越好。

使用方法：
---------------
1. 初始化评估器：
   evaluator = ImageQualityEvaluator()

2. 准备两张图像（路径或PIL对象）：
   real_image = "real.jpg"
   gen_image = "generated.jpg"

3. 计算FID（需要真实图特征和生成图特征）：
   fid = evaluator.compute_fid(
       evaluator.get_inception_features([real_image]),
       evaluator.get_inception_features([gen_image])
   )

4. 计算Inception Score（只需要生成图）：
   is_score = evaluator.compute_is(
       evaluator.get_inception_probs([gen_image])
   )

5. 计算CLIP Score（需要生成图 + 文本描述）：
   clip_score = evaluator.compute_clip_score(
       [gen_image], ["A photo of a cat sitting on the table."]
   )

6. 打印结果：
   print(f"FID: {fid:.4f}, IS: {is_score:.4f}, CLIP: {clip_score:.4f}")
"""

class ImageQualityEvaluator:
    def __init__(self, device=None):
        # 设备设置
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化Inception模型用于FID和IS
        self._init_inception_models()
        
        # 初始化CLIP模型
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 图像预处理管道
        self.inception_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.clip_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def _init_inception_models(self):
        """初始化两个Inception模型分别用于特征提取(FID)和分类概率(IS)"""
        # FID用模型：输出2048维特征
        self.inception_fid = inception_v3(pretrained=True, aux_logits=False)
        self.inception_fid.fc = nn.Identity()  # 移除最后的全连接层
        self.inception_fid = self.inception_fid.to(self.device).eval()
        
        # IS用模型：输出分类logits
        self.inception_is = inception_v3(pretrained=True, aux_logits=False)
        self.inception_is = self.inception_is.to(self.device).eval()

    def _load_images(self, image_list, transform):
        """加载并预处理图像"""
        class ImageDataset(Dataset):
            def __init__(self, images, transform):
                self.images = images
                self.transform = transform

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img = self.images[idx]
                if isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                return self.transform(img)
        
        return ImageDataset(image_list, transform)

    def get_inception_features(self, image_list, batch_size=32):
        """提取Inception-v3特征用于FID计算"""
        dataset = self._load_images(image_list, self.inception_transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        features = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                features.append(self.inception_fid(batch).cpu())
        return torch.cat(features).numpy()

    def get_inception_probs(self, image_list, batch_size=32):
        """获取分类概率用于IS计算"""
        dataset = self._load_images(image_list, self.inception_transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        probs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.inception_is(batch)
                probs.append(F.softmax(logits, dim=1).cpu())
        return torch.cat(probs).numpy()

    def compute_fid(self, real_features, gen_features, eps=1e-6):
        """计算Fréchet Inception Distance"""
        mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
        
        # 计算两个均值向量的平方差
        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # 数值稳定性处理
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        return np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2*covmean)

    def compute_is(self, probs, eps=1e-16):
        """计算Inception Score"""
        # 计算每个图像的KL散度
        marginal_p = np.mean(probs, axis=0)
        kl_div = probs * (np.log(probs + eps) - np.log(marginal_p + eps))
        kl_div = np.sum(kl_div, axis=1)
        
        # 计算指数均值
        return np.exp(np.mean(kl_div))

    def compute_clip_score(self, image_list, text_prompts, batch_size=32):
        """计算CLIP Score"""
        class ImageTextDataset(Dataset):
            def __init__(self, images, texts, transform):
                self.images = images
                self.texts = texts
                self.transform = transform

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img = self.images[idx]
                if isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                return self.transform(img), self.texts[idx]
        
        dataset = ImageTextDataset(image_list, text_prompts, self.clip_transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_score = 0.0
        with torch.no_grad():
            for images, texts in tqdm(loader, desc="CLIP评分"):
                # 处理图像
                images = images.to(self.device)
                
                # 处理文本
                text_inputs = self.clip_processor(
                    text=list(texts), 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # 获取特征
                image_features = self.clip_model.get_image_features(images)
                text_features = self.clip_model.get_text_features(**text_inputs)
                
                # 归一化
                image_features = F.normalize(image_features, p=2, dim=1)
                text_features = F.normalize(text_features, p=2, dim=1)
                
                # 计算相似度
                similarity = (image_features * text_features).sum(dim=1)
                total_score += similarity.clamp(min=0).sum().item()
        
        # 最终分数 = 均值 * 2.5
        return (total_score / len(dataset)) * 2.5
