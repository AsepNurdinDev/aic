import numpy as np
import torch
from ultralytics import YOLO

class RoadGeometryProcessor:
    def __init__(self, model_path: str, device: str = None):
        try:
            self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as e:
            print(f"Failed to load YOLO road geometry model: {e}")
            self.model = None

    def process(self, frame_rgb: np.ndarray) -> dict:
        result_dict = {
            "geometry_available": 0,
            "drivable_fraction": 0.0,
            "lane_fraction": 0.0
        }
        
        if self.model is None:
            return result_dict

        try:
            results = self.model.predict(source=frame_rgb, verbose=False, device=self.device)
            if not results or len(results) == 0:
                return result_dict
                
            result = results[0]
            
            # Extract semantic mask tensor based on prompt contract
            mask_tensor = None
            if hasattr(result, 'semantic_mask') and result.semantic_mask is not None:
                mask_tensor = result.semantic_mask.data
            elif hasattr(result, 'masks') and result.masks is not None:
                mask_tensor = result.masks.data
            
            if mask_tensor is None:
                # No mask found, but model ran successfully
                result_dict["geometry_available"] = 1
                return result_dict
                
            # Convert safely to numpy
            if isinstance(mask_tensor, torch.Tensor):
                mask_np = mask_tensor.squeeze().cpu().numpy()
            else:
                mask_np = np.array(mask_tensor)
                
            # Flatten to compute fractions
            # Assuming mask_np contains class indices: 0: BACKGROUND, 1: DRIVABLE, 2: LANE
            # If it's one-hot or multi-channel, argmax it
            if len(mask_np.shape) == 3:
                # Channels first or last? ultralytics usually (N, H, W) where N is number of classes or instances
                if mask_np.shape[0] == 3: # 3 classes
                    mask_np = np.argmax(mask_np, axis=0)
                else:
                    # instance segmentation? Combine instances
                    # We will sum them up or assume they are binary masks per class
                    # For a single mask with class IDs:
                    pass
            
            total_pixels = mask_np.size
            if total_pixels == 0:
                result_dict["geometry_available"] = 1
                return result_dict
                
            drivable_pixels = np.sum(mask_np == 1)
            lane_pixels = np.sum(mask_np == 2)
            
            result_dict["geometry_available"] = 1
            result_dict["drivable_fraction"] = float(drivable_pixels / total_pixels)
            result_dict["lane_fraction"] = float(lane_pixels / total_pixels)
            
            return result_dict
            
        except Exception as e:
            print(f"Error in road geometry detection: {e}")
            result_dict["geometry_available"] = 0
            return result_dict
