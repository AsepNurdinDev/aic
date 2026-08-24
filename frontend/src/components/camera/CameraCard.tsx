import React, { useRef } from "react";
import { Camera } from "../../types/camera";

interface CameraCardProps {
    camera: Camera;
    onUpload: (cameraId: string, file: File) => void;
}

export const CameraCard: React.FC<CameraCardProps> = ({ camera, onUpload }) => {
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            onUpload(camera.id, e.target.files[0]);
        }
    };

    return (
        <div className="bg-gray-900 rounded-lg border border-gray-800 flex flex-col overflow-hidden h-64">
            <div className="p-2 bg-gray-800 flex justify-between items-center border-b border-gray-700">
                <h3 className="text-sm font-bold text-gray-200">{camera.title}</h3>
                <span className={`text-xs px-2 py-0.5 rounded font-bold ${
                    camera.status === 'CONNECTED' ? 'bg-green-900 text-green-300' :
                    camera.status === 'UPLOADING' ? 'bg-blue-900 text-blue-300' :
                    'bg-gray-700 text-gray-300'
                }`}>
                    {camera.mode === 'LIVE' ? camera.status : 'UPLOAD MODE'}
                </span>
            </div>
            
            <div className="flex-grow bg-black relative flex items-center justify-center">
                {camera.mode === 'LIVE' ? (
                    <div className="text-gray-600 text-sm flex flex-col items-center">
                        <span className="text-2xl mb-2">📡</span>
                        Waiting for Live Stream...
                    </div>
                ) : (
                    <div className="text-center p-4">
                        {camera.status === 'UPLOADING' ? (
                            <div className="text-blue-400 animate-pulse">Uploading...</div>
                        ) : (
                            <>
                                <p className="text-gray-500 text-sm mb-4">Select a video file to simulate camera feed.</p>
                                <button 
                                    onClick={() => fileInputRef.current?.click()}
                                    className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded text-sm font-bold transition-colors"
                                >
                                    Choose Video
                                </button>
                                <input 
                                    type="file" 
                                    className="hidden" 
                                    ref={fileInputRef} 
                                    accept="video/mp4,video/webm" 
                                    onChange={handleFileChange}
                                />
                            </>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};
