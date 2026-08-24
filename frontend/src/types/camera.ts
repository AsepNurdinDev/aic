export type CameraMode = "LIVE" | "UPLOAD";

export type CameraStatus =
    | "CONNECTED"
    | "DISCONNECTED"
    | "UPLOADING"
    | "PROCESSING"
    | "READY"
    | "ERROR";

export type Camera = {
    id: string;
    title: string;
    description: string;
    mode: CameraMode;
    status: CameraStatus;
    streamUrl?: string;
    uploadedVideoUrl?: string;
};
