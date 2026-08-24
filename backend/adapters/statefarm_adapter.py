from typing import Dict, Any, List
from fusion.feature_schema import StateFarmOutput

class StateFarmAdapter:
    """
    Adapter to convert ConvNeXt-Tiny StateFarm probabilities into standardized StateFarmOutput.
    """
    
    CLASSES = [
        "drinking",
        "normal",
        "phone",
        "radio",
        "reaching_behind",
        "talking_passenger",
        "texting"
    ]
    
    @staticmethod
    def adapt(probs: List[float]) -> StateFarmOutput:
        if not probs or len(probs) != 7:
            return StateFarmOutput(statefarm_available=0)
            
        max_prob = max(probs)
        pred_idx = probs.index(max_prob)
        predicted_class = StateFarmAdapter.CLASSES[pred_idx]
        
        return StateFarmOutput(
            statefarm_available=1,
            sf_drinking_prob=probs[0],
            sf_normal_prob=probs[1],
            sf_phone_prob=probs[2],
            sf_radio_prob=probs[3],
            sf_reaching_prob=probs[4],
            sf_passenger_prob=probs[5],
            sf_texting_prob=probs[6],
            sf_predicted_class=predicted_class,
            sf_confidence=max_prob
        )
