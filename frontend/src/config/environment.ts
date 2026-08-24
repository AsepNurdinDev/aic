export const env = {
    API_BASE_URL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
    DRIVER_STREAM_URL: import.meta.env.VITE_DRIVER_STREAM_URL || "",
    FRONT_STREAM_URL: import.meta.env.VITE_FRONT_STREAM_URL || "",
    CABIN_STREAM_URL: import.meta.env.VITE_CABIN_STREAM_URL || "",
    MOCK_MODE: import.meta.env.VITE_MOCK_MODE === "true"
};
