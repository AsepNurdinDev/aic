import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

class StateFarmProcessor:
    def __init__(self, model_path: str, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load ConvNeXt-Tiny architecture
        self.model = models.convnext_tiny(weights=None)
        
        # Modify the classifier for 7 classes
        # ConvNeXt classifier is typically a Sequential with a LayerNorm, Flatten, and Linear
        num_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(num_features, 7)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Transforms (standard ImageNet normalization typically used for ConvNeXt)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Based on typical ImageFolder training (alphabetical)
        self.classes = [
            "drinking",
            "normal",
            "phone",
            "radio",
            "reaching_behind",
            "talking_passenger",
            "texting"
        ]

    def process(self, frame_rgb: np.ndarray) -> dict:
        try:
            # frame_rgb is HxWxC
            input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                
            # Map probabilities to specific variables
            prob_dict = {cls: float(p) for cls, p in zip(self.classes, probs)}
            
            max_prob = float(np.max(probs))
            
            # Distraction is 1.0 - normal_prob
            distraction_prob = 1.0 - prob_dict["normal"]
            
            result = {
                "statefarm_available": 1,
                "sf_drinking_prob": prob_dict["drinking"],
                "sf_normal_prob": prob_dict["normal"],
                "sf_phone_prob": prob_dict["phone"],
                "sf_radio_prob": prob_dict["radio"],
                "sf_reaching_prob": prob_dict["reaching_behind"],
                "sf_passenger_prob": prob_dict["talking_passenger"],
                "sf_texting_prob": prob_dict["texting"],
                "sf_behavior_max_prob": max_prob,
                "sf_distraction_prob": distraction_prob
            }
            return result
            
        except Exception as e:
            return {
                "statefarm_available": 0,
                "sf_drinking_prob": 0.0,
                "sf_normal_prob": 0.0,
                "sf_phone_prob": 0.0,
                "sf_radio_prob": 0.0,
                "sf_reaching_prob": 0.0,
                "sf_passenger_prob": 0.0,
                "sf_texting_prob": 0.0,
                "sf_behavior_max_prob": 0.0,
                "sf_distraction_prob": 0.0
            }
