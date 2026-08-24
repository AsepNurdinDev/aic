import React from "react";
import { DecisionState, Severity } from "../../types/decision";

interface DecisionSummaryProps {
    decision: DecisionState;
}

export const DecisionSummary: React.FC<DecisionSummaryProps> = ({ decision }) => {
    const getSeverityColor = (severity: Severity) => {
        switch (severity) {
            case "SAFE": return "text-green-500 bg-green-500/10";
            case "CAUTION": return "text-yellow-500 bg-yellow-500/10";
            case "HIGH": return "text-orange-500 bg-orange-500/10";
            case "CRITICAL": return "text-red-500 bg-red-500/10";
            case "DEGRADED": return "text-gray-400 bg-gray-400/10";
            default: return "text-gray-500 bg-gray-500/10";
        }
    };

    return (
        <div className="bg-gray-900 p-4 rounded-lg border border-gray-800">
            <h2 className="text-sm text-gray-400 font-bold mb-4 uppercase tracking-wider">Decision AI Summary</h2>
            
            <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-800 rounded p-3">
                    <div className="text-xs text-gray-400 mb-1">MODE</div>
                    <div className={`font-bold ${decision.decision_mode === 'FULL' ? 'text-blue-400' : 'text-yellow-500'}`}>
                        {decision.decision_mode}
                    </div>
                </div>
                
                <div className="bg-gray-800 rounded p-3">
                    <div className="text-xs text-gray-400 mb-1">EVENTS</div>
                    <div className="font-bold text-white">
                        {decision.observed_event_count} Active
                    </div>
                </div>
                
                <div className="col-span-2">
                    <div className="text-xs text-gray-400 mb-1">SEVERITY</div>
                    <div className={`text-xl font-bold p-2 rounded text-center ${getSeverityColor(decision.severity)}`}>
                        {decision.severity}
                    </div>
                </div>
                
                <div className="col-span-2">
                    <div className="text-xs text-gray-400 mb-1">ACTION REQUIRED</div>
                    <div className="font-bold text-white bg-red-900/30 p-2 rounded text-center border border-red-900/50">
                        {decision.action.replace(/_/g, ' ')}
                    </div>
                </div>
            </div>
        </div>
    );
};
