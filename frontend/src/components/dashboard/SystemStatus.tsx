import React from "react";
import { SystemState } from "../../types/system";

interface SystemStatusProps {
    systemState: SystemState;
}

export const SystemStatus: React.FC<SystemStatusProps> = ({ systemState }) => {
    const { models, connection } = systemState;
    
    const StatusBadge = ({ status }: { status: "READY" | "ERROR" }) => (
        <span className={`text-xs font-bold px-2 py-1 rounded ${status === 'READY' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
            {status}
        </span>
    );

    return (
        <div className="bg-gray-900 p-4 rounded-lg border border-gray-800 flex flex-col">
            <h2 className="text-sm text-gray-400 font-bold mb-4 uppercase tracking-wider">System Status</h2>
            <div className="space-y-3 flex-grow">
                <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-300">FL3D Drowsiness</span>
                    <StatusBadge status={models.drowsiness} />
                </div>
                <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-300">State Farm</span>
                    <StatusBadge status={models.statefarm} />
                </div>
                <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-300">Road Object</span>
                    <StatusBadge status={models.road_object} />
                </div>
                <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-300">Road Geometry</span>
                    <StatusBadge status={models.road_geometry} />
                </div>
                <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-300">Decision AI</span>
                    <StatusBadge status={models.decision} />
                </div>
            </div>
            
            <div className="mt-4 pt-4 border-t border-gray-800">
                <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-300">Connection</span>
                    <span className={`text-sm font-bold ${
                        connection === 'CONNECTED' ? 'text-green-500' : 
                        connection === 'DEGRADED' ? 'text-yellow-500' : 'text-red-500'
                    }`}>
                        {connection}
                    </span>
                </div>
            </div>
        </div>
    );
};
