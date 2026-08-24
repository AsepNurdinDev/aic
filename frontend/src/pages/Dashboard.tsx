import React, { useState, useEffect } from "react";
import { Header } from "../components/dashboard/Header";
import { CameraGrid } from "../components/camera/CameraGrid";
import { DecisionSummary } from "../components/dashboard/DecisionSummary";
import { EventPanel } from "../components/dashboard/EventPanel";
import { SystemStatus } from "../components/dashboard/SystemStatus";
import { WarningBanner } from "../components/dashboard/WarningBanner";
import { MockAPI } from "../services/mockApi";
import { Camera } from "../types/camera";
import { DecisionState } from "../types/decision";
import { SystemState } from "../types/system";

export const Dashboard: React.FC = () => {
    const [cameras, setCameras] = useState<Camera[]>([
        { id: "cam1", title: "DRIVER CAMERA (Drowsiness / StateFarm)", description: "Monitors driver face and behavior", mode: "UPLOAD", status: "DISCONNECTED" },
        { id: "cam2", title: "FRONT CAMERA (Road Risk)", description: "Monitors forward road objects and geometry", mode: "LIVE", status: "CONNECTED" },
        { id: "cam3", title: "CABIN CAMERA (Overall Context)", description: "Monitors passenger and cabin activities", mode: "LIVE", status: "CONNECTED" }
    ]);

    const [decision, setDecision] = useState<DecisionState | null>(null);
    const [isDemoRunning, setIsDemoRunning] = useState(false);
    const [fps, setFps] = useState(30.0);

    const systemState: SystemState = {
        models: {
            drowsiness: decision?.availability.drowsiness ? "READY" : "ERROR",
            statefarm: decision?.availability.statefarm ? "READY" : "ERROR",
            road_object: decision?.availability.road ? "READY" : "ERROR",
            road_geometry: decision?.availability.road ? "READY" : "ERROR",
            decision: "READY"
        },
        connection: decision?.decision_mode === "DEGRADED" ? "DEGRADED" : (isDemoRunning ? "CONNECTED" : "OFFLINE"),
        fps: fps
    };

    useEffect(() => {
        const unsubscribe = MockAPI.subscribe((newDecision) => {
            setDecision(newDecision);
            setFps(28 + Math.random() * 4); // Simulate 28-32 FPS fluctuation
        });
        return () => unsubscribe();
    }, []);

    const handleToggleDemo = () => {
        if (isDemoRunning) {
            MockAPI.stopDemo();
            setIsDemoRunning(false);
            setDecision(null);
            setFps(0);
        } else {
            MockAPI.startDemo();
            setIsDemoRunning(true);
        }
    };

    const handleToggleMode = (cameraId: string) => {
        setCameras(cams => cams.map(c => {
            if (c.id === cameraId) {
                return { 
                    ...c, 
                    mode: c.mode === 'LIVE' ? 'UPLOAD' : 'LIVE',
                    status: c.mode === 'LIVE' ? 'DISCONNECTED' : 'CONNECTED'
                };
            }
            return c;
        }));
    };

    const handleUpload = async (cameraId: string, file: File) => {
        setCameras(cams => cams.map(c => c.id === cameraId ? { ...c, status: 'UPLOADING' } : c));
        await MockAPI.uploadVideo(cameraId, file);
        setCameras(cams => cams.map(c => c.id === cameraId ? { ...c, status: 'READY' } : c));
    };

    return (
        <div className="min-h-screen bg-black text-white flex flex-col font-sans">
            {decision && <WarningBanner decision={decision} />}
            <Header connectionStatus={systemState.connection} fps={systemState.fps} />
            
            <main className="flex-grow p-4 overflow-hidden flex flex-col space-y-4">
                <div className="flex justify-between items-center bg-gray-900 p-3 rounded-lg border border-gray-800">
                    <div>
                        <h2 className="text-lg font-bold">Live Monitoring Dashboard</h2>
                        <p className="text-gray-400 text-sm">Reviewing 3 concurrent perception streams</p>
                    </div>
                    <button 
                        onClick={handleToggleDemo}
                        className={`px-6 py-2 rounded font-bold transition-colors ${
                            isDemoRunning ? 'bg-red-600 hover:bg-red-500' : 'bg-green-600 hover:bg-green-500'
                        }`}
                    >
                        {isDemoRunning ? 'STOP DEMO MOCK' : 'START DEMO MOCK'}
                    </button>
                </div>

                <div className="grid grid-cols-1 xl:grid-cols-4 gap-4 flex-grow">
                    <div className="xl:col-span-3 space-y-4">
                        <CameraGrid 
                            cameras={cameras} 
                            onToggleMode={handleToggleMode} 
                            onUpload={handleUpload} 
                        />
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {decision ? (
                                <>
                                    <DecisionSummary decision={decision} />
                                    <EventPanel decision={decision} />
                                </>
                            ) : (
                                <div className="col-span-2 bg-gray-900 p-8 rounded-lg border border-gray-800 text-center text-gray-500">
                                    <p>Waiting for Decision AI data stream...</p>
                                    <p className="text-sm mt-2">Click 'START DEMO MOCK' to simulate incoming data.</p>
                                </div>
                            )}
                        </div>
                    </div>
                    
                    <div className="xl:col-span-1">
                        <SystemStatus systemState={systemState} />
                    </div>
                </div>
            </main>
        </div>
    );
};
