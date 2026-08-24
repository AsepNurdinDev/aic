export type DecisionMode = "FULL" | "DEGRADED";

export type Severity = "SAFE" | "CAUTION" | "HIGH" | "CRITICAL" | "DEGRADED";

export type Action =
    | "NONE"
    | "DROWSINESS_WARNING"
    | "DISTRACTION_WARNING"
    | "ROAD_WARNING"
    | "URGENT_WARNING"
    | "DEGRADED_WARNING";

export type EventState = {
    probability: number | null;
    threshold: number | null;
    active: boolean | null;
    state?: string;
};

export type DecisionState = {
    timestamp: string | number;
    decision_mode: DecisionMode;

    availability: {
        drowsiness: boolean;
        statefarm: boolean;
        road: boolean;
    };

    events: {
        drowsiness: EventState;
        phone_use: EventState;
        texting: EventState;
        drinking: EventState;
        radio_operation: EventState;
        reaching_behind: EventState;
        talking_passenger: EventState;
        road_risk: EventState;
        [key: string]: EventState;
    };

    normal: boolean;
    observed_event_count: number;
    severity: Severity;
    action: Action;
    warning_message?: string;
    warning_id?: string;
};
