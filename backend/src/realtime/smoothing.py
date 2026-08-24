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
            raw_active = info.get("active")
            
            if raw_active is None:
                smoothed_events[event] = info.copy()
                continue
            
            # Initialize counters if not present
            if event not in self.current_state:
                self.current_state[event] = False
                self.active_consecutive[event] = 0
                self.inactive_consecutive[event] = 0
                
            if raw_active:
                self.active_consecutive[event] += 1
                # Leaky bucket: forgive intermittent noise instead of hard reset
                self.inactive_consecutive[event] = max(0, self.inactive_consecutive[event] - 1)
            else:
                self.inactive_consecutive[event] += 1
                # Leaky bucket: forgive intermittent noise instead of hard reset
                self.active_consecutive[event] = max(0, self.active_consecutive[event] - 1)
                
            # State transitions
            if not self.current_state[event] and self.active_consecutive[event] >= self.confirm_frames:
                self.current_state[event] = True
                self.inactive_consecutive[event] = 0
            elif self.current_state[event] and self.inactive_consecutive[event] >= self.clear_frames:
                self.current_state[event] = False
                self.active_consecutive[event] = 0
                
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
        if decision.get("decision_mode") == "FULL":
            decision["normal"] = (observed_count == 0)
        else:
            decision["normal"] = False if observed_count > 0 else None
        
        return decision
