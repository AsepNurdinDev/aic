import numpy as np
from ultralytics import YOLO

class RoadObjectProcessor:
    def __init__(self, model_path: str, device: str = None):
        try:
            self.model = YOLO(model_path)
            self.device = device if device else "cuda"
            self.model.to(self.device)
        except Exception as e:
            print(f"Failed to load YOLO road object model: {e}")
            self.model = None

    def process(self, frame_rgb: np.ndarray) -> dict:
        result_dict = {
            "road_available": 0,
            "object_count": 0,
            "car_count": 0,
            "truck_count": 0,
            "bus_count": 0,
            "two_wheeler_count": 0,
            "person_count": 0,
            "max_object_confidence": 0.0,
            "nearest_object_class": "none",
            "nearest_object_confidence": 0.0,
            "nearest_object_area_ratio": 0.0
        }
        
        if self.model is None:
            return result_dict

        try:
            results = self.model.predict(source=frame_rgb, verbose=False, device=self.device)
            if not results or len(results) == 0:
                return result_dict
                
            result = results[0]
            boxes = result.boxes
            
            if len(boxes) == 0:
                result_dict["road_available"] = 1
                return result_dict
                
            img_h, img_w = frame_rgb.shape[:2]
            img_area = img_w * img_h
            
            names = self.model.names
            
            max_conf = 0.0
            max_area_ratio = 0.0
            nearest_cls = "none"
            nearest_conf = 0.0
            
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = names[cls_id].lower()
                
                # Bounding box xywh
                x_c, y_c, w, h = box.xywh[0].cpu().numpy()
                area_ratio = (w * h) / img_area
                
                result_dict["object_count"] += 1
                
                if "car" in cls_name:
                    result_dict["car_count"] += 1
                elif "truck" in cls_name:
                    result_dict["truck_count"] += 1
                elif "bus" in cls_name:
                    result_dict["bus_count"] += 1
                elif "motorcycle" in cls_name or "bicycle" in cls_name or "bike" in cls_name:
                    result_dict["two_wheeler_count"] += 1
                elif "person" in cls_name:
                    result_dict["person_count"] += 1
                    
                if conf > max_conf:
                    max_conf = conf
                    
                if area_ratio > max_area_ratio:
                    max_area_ratio = area_ratio
                    nearest_cls = cls_name
                    nearest_conf = conf
                    
            result_dict["road_available"] = 1
            result_dict["max_object_confidence"] = max_conf
            result_dict["nearest_object_class"] = nearest_cls
            result_dict["nearest_object_confidence"] = nearest_conf
            result_dict["nearest_object_area_ratio"] = max_area_ratio
            
            return result_dict
            
        except Exception as e:
            print(f"Error in road object detection: {e}")
            result_dict["road_available"] = 0
            return result_dict
