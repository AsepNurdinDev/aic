import React, { useEffect, useState } from "react";
import { DecisionState } from "../../types/decision";

interface WarningBannerProps {
    decision: DecisionState;
}

export const WarningBanner: React.FC<WarningBannerProps> = ({ decision }) => {
    const [show, setShow] = useState(false);
    
    useEffect(() => {
        if (decision.severity === "HIGH" || decision.severity === "CRITICAL" || decision.action !== "NONE") {
            setShow(true);
        } else {
            setShow(false);
        }
    }, [decision.severity, decision.action]);

    if (!show) return null;

    const isCritical = decision.severity === "CRITICAL";

    return (
        <div className={`fixed top-4 left-1/2 transform -translate-x-1/2 z-50 p-4 rounded shadow-2xl transition-all w-3/4 max-w-2xl text-center border-2 animate-pulse ${
            isCritical ? "bg-red-600 border-red-900 text-white" : "bg-orange-500 border-orange-800 text-white"
        }`}>
            <h2 className="text-2xl font-black uppercase tracking-widest">{decision.action.replace(/_/g, ' ')}</h2>
            <p className="mt-1 text-sm opacity-90">{decision.warning_message || "Immediate action required by the driver."}</p>
        </div>
    );
};
