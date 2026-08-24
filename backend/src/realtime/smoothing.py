class EventSmoother:
    def __init__(self, confirm_frames=3, clear_frames=5):
        self.confirm_frames = confirm_frames
        self.clear_frames = clear_frames
        
        # State tracking per event
        self.active_consecutive = {}
        self.inactive_consecutive = {}
        self.current_state = {}

    def smooth(self, raw_events: dict) -> dict:
        smoothed_events = {}
        
        for event, info in raw_events.items():
            raw_active = info["active"]
            
            # Initialize counters if not present
            if event not in self.current_state:
                self.current_state[event] = False
                self.active_consecutive[event] = 0
                self.inactive_consecutive[event] = 0
                
            if raw_active:
                self.active_consecutive[event] += 1
                self.inactive_consecutive[event] = 0
            else:
                self.inactive_consecutive[event] += 1
                self.active_consecutive[event] = 0
                
            # State transitions
            if not self.current_state[event] and self.active_consecutive[event] >= self.confirm_frames:
                self.current_state[event] = True
            elif self.current_state[event] and self.inactive_consecutive[event] >= self.clear_frames:
                self.current_state[event] = False
                
            # Create a copy and update active state
            smoothed_info = info.copy()
            smoothed_info["active"] = self.current_state[event]
            smoothed_info["state"] = "ACTIVE" if self.current_state[event] else "INACTIVE"
            smoothed_events[event] = smoothed_info
            
        return smoothed_events

class DecisionSmoother:
    def __init__(self, confirm_frames=3, clear_frames=5):
        self.event_smoother = EventSmoother(confirm_frames, clear_frames)
        
    def process(self, decision: dict) -> dict:
        raw_events = decision.get("events", {})
        smoothed_events = self.event_smoother.smooth(raw_events)
        
        decision["events"] = smoothed_events
        
        # Re-evaluate observed count and normality based on smoothed events
        observed_count = sum(1 for e in smoothed_events.values() if e.get("active", False))
        decision["observed_event_count"] = observed_count
        decision["normal"] = (observed_count == 0)
        
        return decision
