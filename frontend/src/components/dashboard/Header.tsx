import React from "react";
import { SystemConnection } from "../../types/system";

interface HeaderProps {
    connectionStatus: SystemConnection;
    fps: number;
}

export const Header: React.FC<HeaderProps> = ({ connectionStatus, fps }) => {
    const getStatusColor = (status: SystemConnection) => {
        switch (status) {
            case "CONNECTED": return "text-green-500";
            case "DEGRADED": return "text-yellow-500";
            case "OFFLINE": return "text-red-500";
            default: return "text-gray-500";
        }
    };

    return (
        <header className="flex justify-between items-center p-4 bg-gray-900 text-white border-b border-gray-800">
            <div className="flex items-center space-x-4">
                <h1 className="text-2xl font-bold tracking-wider">DRIVER SAFETY AI</h1>
                <span className="px-2 py-1 text-xs font-semibold bg-gray-800 rounded">DEMO / MOCK</span>
            </div>
            
            <div className="flex items-center space-x-6">
                <div className="flex items-center space-x-2">
                    <span className="text-gray-400 text-sm">System Status:</span>
                    <span className={`font-bold ${getStatusColor(connectionStatus)}`}>
                        {connectionStatus}
                    </span>
                </div>
                <div className="flex items-center space-x-2">
                    <span className="text-gray-400 text-sm">FPS:</span>
                    <span className="font-mono text-blue-400">{fps.toFixed(1)}</span>
                </div>
            </div>
        </header>
    );
};
