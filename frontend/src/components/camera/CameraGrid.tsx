import React from "react";
import { Camera } from "../../types/camera";
import { CameraCard } from "./CameraCard";

interface CameraGridProps {
    cameras: Camera[];
    onUpload: (cameraId: string, file: File) => void;
    onToggleMode: (cameraId: string) => void;
}

export const CameraGrid: React.FC<CameraGridProps> = ({ cameras, onUpload, onToggleMode }) => {
    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {cameras.map(camera => (
                <div key={camera.id} className="flex flex-col">
                    <CameraCard camera={camera} onUpload={onUpload} />
                    <button 
                        onClick={() => onToggleMode(camera.id)}
                        className="mt-2 text-xs text-gray-500 hover:text-white transition-colors text-right"
                    >
                        Switch to {camera.mode === 'LIVE' ? 'Upload' : 'Live'} Mode
                    </button>
                </div>
            ))}
        </div>
    );
};
