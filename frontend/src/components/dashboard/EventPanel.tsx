import React from "react";
import { DecisionState } from "../../types/decision";

interface EventPanelProps {
    decision: DecisionState;
}

export const EventPanel: React.FC<EventPanelProps> = ({ decision }) => {
    const renderProgressBar = (value: number | null, threshold: number | null, active: boolean | null) => {
        if (value === null) return <div className="h-2 bg-gray-700 rounded mt-1 w-full" />;
        
        const pct = Math.min(Math.max(value * 100, 0), 100);
        const thPct = threshold !== null ? Math.min(Math.max(threshold * 100, 0), 100) : null;
        
        return (
            <div className="relative h-2 bg-gray-700 rounded mt-1 w-full overflow-hidden">
                <div 
                    className={`absolute top-0 left-0 h-full ${active ? 'bg-red-500' : 'bg-blue-500'}`} 
                    style={{ width: `${pct}%` }} 
                />
                {thPct !== null && (
                    <div 
                        className="absolute top-0 h-full w-0.5 bg-yellow-400 z-10" 
                        style={{ left: `${thPct}%` }}
                    />
                )}
            </div>
        );
    };

    const eventsList = [
        { id: 'drowsiness', label: 'Drowsiness' },
        { id: 'phone_use', label: 'Phone Use' },
        { id: 'texting', label: 'Texting' },
        { id: 'drinking', label: 'Drinking' },
        { id: 'radio_operation', label: 'Radio Ops' },
        { id: 'reaching_behind', label: 'Reaching' },
        { id: 'talking_passenger', label: 'Talking' },
        { id: 'road_risk', label: 'Road Risk' }
    ];

    return (
        <div className="bg-gray-900 p-4 rounded-lg border border-gray-800">
            <h2 className="text-sm text-gray-400 font-bold mb-4 uppercase tracking-wider">Perception Events</h2>
            <div className="space-y-4">
                {eventsList.map(evt => {
                    const data = decision.events[evt.id];
                    if (!data) return null;
                    return (
                        <div key={evt.id} className="text-sm">
                            <div className="flex justify-between text-gray-300">
                                <span>{evt.label}</span>
                                <span className={data.active ? 'text-red-400 font-bold' : 'text-gray-500'}>
                                    {data.probability !== null ? (data.probability * 100).toFixed(0) + '%' : 'N/A'}
                                </span>
                            </div>
                            {renderProgressBar(data.probability, data.threshold, data.active)}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};
