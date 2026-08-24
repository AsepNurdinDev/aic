import { DecisionState } from "../types/decision";

let mockInterval: any = null;
let listeners: ((decision: DecisionState) => void)[] = [];

// Base dummy state
const generateMockDecision = (): DecisionState => {
    return {
        timestamp: new Date().toISOString(),
        decision_mode: "FULL",
        availability: {
            drowsiness: true,
            statefarm: true,
            road: true
        },
        events: {
            drowsiness: { probability: Math.random() * 0.1, threshold: 0.05, active: false, state: "INACTIVE" },
            phone_use: { probability: Math.random() * 0.2, threshold: 0.91, active: false, state: "INACTIVE" },
            texting: { probability: 0.0, threshold: 0.40, active: false, state: "INACTIVE" },
            drinking: { probability: 0.0, threshold: 0.19, active: false, state: "INACTIVE" },
            radio_operation: { probability: 0.01, threshold: 0.05, active: false, state: "INACTIVE" },
            reaching_behind: { probability: 0.0, threshold: 0.40, active: false, state: "INACTIVE" },
            talking_passenger: { probability: 0.1, threshold: 0.47, active: false, state: "INACTIVE" },
            road_risk: { probability: Math.random() * 0.05, threshold: 0.05, active: false, state: "INACTIVE" }
        },
        normal: true,
        observed_event_count: 0,
        severity: "SAFE",
        action: "NONE"
    };
};

export const MockAPI = {
    startDemo: () => {
        console.log("[MOCK] Starting DEMO mode");
        if (mockInterval) clearInterval(mockInterval);
        
        mockInterval = setInterval(() => {
            const decision = generateMockDecision();
            
            // Randomly trigger an event for demo purposes
            if (Math.random() > 0.8) {
                decision.events.drowsiness.probability = 0.95;
                decision.events.drowsiness.active = true;
                decision.events.drowsiness.state = "ACTIVE";
                decision.severity = "HIGH";
                decision.action = "DROWSINESS_WARNING";
                decision.observed_event_count = 1;
                decision.normal = false;
            } else if (Math.random() > 0.9) {
                decision.decision_mode = "DEGRADED";
                decision.availability.drowsiness = false;
                decision.severity = "DEGRADED";
                decision.action = "DEGRADED_WARNING";
            }
            
            listeners.forEach(cb => cb(decision));
        }, 1000);
    },
    
    stopDemo: () => {
        console.log("[MOCK] Stopping DEMO mode");
        if (mockInterval) clearInterval(mockInterval);
    },
    
    subscribe: (callback: (decision: DecisionState) => void) => {
        listeners.push(callback);
        return () => {
            listeners = listeners.filter(cb => cb !== callback);
        };
    },
    
    uploadVideo: async (cameraId: string, file: File) => {
        console.log(`[MOCK] Uploading video for camera ${cameraId}`, file.name);
        return new Promise(resolve => {
            setTimeout(() => {
                resolve({ job_id: `mock-job-${Date.now()}` });
            }, 2000);
        });
    }
};
