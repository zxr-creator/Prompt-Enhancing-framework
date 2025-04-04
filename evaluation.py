"""
ImageQualityEvaluator: Image Quality Evaluation Tool

1. FID (Fréchet Inception Distance): Measures the distribution difference between generated and real images in the feature space. Lower is better.
2. IS (Inception Score): Measures whether the generated image is semantically clear and class-diverse. Higher is better.
3. CLIP Score: Measures semantic similarity between the generated image and a text description. Higher is better.

Usage:
---------------
1. Initialize the evaluator:
   evaluator = ImageQualityEvaluator()

2. Prepare two images (file path or PIL Image object):
   real_image = "real.jpg"
   gen_image = "generated.jpg"

3. Compute FID (requires features of real and generated images):
   fid = evaluator.compute_fid(
       evaluator.get_inception_features([real_image]),
       evaluator.get_inception_features([gen_image])
   )

4. Compute Inception Score (only needs generated images):
   is_score = evaluator.compute_is(
       evaluator.get_inception_probs([gen_image])
   )

5. Compute CLIP Score (requires generated image + text description):
   clip_score = evaluator.compute_clip_score(
       [gen_image], ["A photo of a cat sitting on the table."]
   )

6. Print results:
   print(f"FID: {fid:.4f}, IS: {is_score:.4f}, CLIP: {clip_score:.4f}")
"""

class ImageQualityEvaluator:
    def __init__(self, device=None):
        # Device setup
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Inception models for FID and IS
        self._init_inception_models()
        
        # Initialize CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Image preprocessing pipeline
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
        """Initialize two Inception models for feature extraction (FID) and classification probabilities (IS)"""
        # Model for FID: outputs 2048-d features
        self.inception_fid = inception_v3(pretrained=True, aux_logits=False)
        self.inception_fid.fc = nn.Identity()  # Remove the final fully connected layer
        self.inception_fid = self.inception_fid.to(self.device).eval()
        
        # Model for IS: outputs classification logits
        self.inception_is = inception_v3(pretrained=True, aux_logits=False)
        self.inception_is = self.inception_is.to(self.device).eval()

    def _load_images(self, image_list, transform):
        """Load and preprocess images"""
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
        """Extract Inception-v3 features for FID computation"""
        dataset = self._load_images(image_list, self.inception_transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        features = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                features.append(self.inception_fid(batch).cpu())
        return torch.cat(features).numpy()

    def get_inception_probs(self, image_list, batch_size=32):
        """Get classification probabilities for IS computation"""
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
        """Compute Fréchet Inception Distance"""
        mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
        
        # Compute squared difference between means
        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical stability handling
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        return np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2*covmean)

    def compute_is(self, probs, eps=1e-16):
        """Compute Inception Score"""
        # Compute KL divergence for each image
        marginal_p = np.mean(probs, axis=0)
        kl_div = probs * (np.log(probs + eps) - np.log(marginal_p + eps))
        kl_div = np.sum(kl_div, axis=1)
        
        # Compute exponential of mean KL divergence
        return np.exp(np.mean(kl_div))

    def compute_clip_score(self, image_list, text_prompts, batch_size=32):
        """Compute CLIP Score"""
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
            for images, texts in tqdm(loader, desc="CLIP Score"):
                # Process images
                images = images.to(self.device)
                
                # Process texts
                text_inputs = self.clip_processor(
                    text=list(texts), 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Get features
                image_features = self.clip_model.get_image_features(images)
                text_features = self.clip_model.get_text_features(**text_inputs)
                
                # Normalize
                image_features = F.normalize(image_features, p=2, dim=1)
                text_features = F.normalize(text_features, p=2, dim=1)
                
                # Compute similarity
                similarity = (image_features * text_features).sum(dim=1)
                total_score += similarity.clamp(min=0).sum().item()
        
        # Final score = average * 2.5
        return (total_score / len(dataset)) * 2.5
