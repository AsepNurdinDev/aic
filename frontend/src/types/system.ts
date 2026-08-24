export type ModelStatus = "READY" | "ERROR";

export type SystemConnection = "CONNECTED" | "DEGRADED" | "OFFLINE";

export type SystemState = {
    models: {
        drowsiness: ModelStatus;
        statefarm: ModelStatus;
        road_object: ModelStatus;
        road_geometry: ModelStatus;
        decision: ModelStatus;
    };
    connection: SystemConnection;
    fps: number;
};
