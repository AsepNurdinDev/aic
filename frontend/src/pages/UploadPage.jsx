import { useState, useRef, useEffect } from "react";
import "../App.css";

function UploadPage({ onClose, onUpload }) {
const [driverMode, setDriverMode] = useState(null); // 'file' | 'camera' | null
const [driverFile, setDriverFile] = useState(null);
const [driverStream, setDriverStream] = useState(null);

const [roadFile, setRoadFile] = useState(null);

const [showChoice, setShowChoice] = useState(false);
const [cameraError, setCameraError] = useState(null);

const driverInputRef = useRef(null);
const roadInputRef = useRef(null);
const driverPreviewRef = useRef(null);

// Menandai apakah stream kamera sudah "diserahkan" ke HomePage,
// supaya tidak dimatikan otomatis saat UploadPage unmount setelah Upload berhasil.
const streamHandedOff = useRef(false);

// Pasang live stream ke elemen <video> preview di dalam box upload
useEffect(() => {
   if (driverMode === "camera" && driverStream && driverPreviewRef.current) {
   driverPreviewRef.current.srcObject = driverStream;
   }
}, [driverMode, driverStream]);

// Matikan kamera kalau popup ditutup TANPA klik Upload (misal Cancel)
useEffect(() => {
   return () => {
   if (driverStream && !streamHandedOff.current) {
      driverStream.getTracks().forEach((t) => t.stop());
   }
   };
}, [driverStream]);

const handleDriverFileChange = (e) => {
   if (e.target.files.length > 0) {
   // kalau sebelumnya sempat pilih live camera, matikan dulu
   if (driverStream) {
      driverStream.getTracks().forEach((t) => t.stop());
      setDriverStream(null);
   }
   setDriverFile(e.target.files[0]);
   setDriverMode("file");
   }
};

const handleRoadFileChange = (e) => {
   if (e.target.files.length > 0) setRoadFile(e.target.files[0]);
};

const handleDriverBoxClick = () => setShowChoice(true);
const handleRoadBoxClick = () => roadInputRef.current.click();

const handleChooseUpload = () => {
   setShowChoice(false);
   driverInputRef.current.click();
};

const handleChooseCamera = async () => {
   setCameraError(null);
   try {
   const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
   });
   setDriverFile(null);
   setDriverStream(stream);
   setDriverMode("camera");
   setShowChoice(false); // langsung balik ke popup upload utama
   } catch (err) {
   setCameraError(
      "Tidak bisa mengakses kamera. Periksa izin kamera pada browser."
   );
   }
};

const handleUploadClick = () => {
   streamHandedOff.current = true; // serahkan stream ke HomePage, jangan di-stop
   onUpload({
   driver:
      driverMode === "camera"
         ? { type: "camera", stream: driverStream }
         : driverMode === "file"
         ? { type: "file", file: driverFile }
         : null,
   road: roadFile ? { type: "file", file: roadFile } : null,
   });
};

const driverReady =
   driverMode === "file"
   ? !!driverFile
   : driverMode === "camera"
   ? !!driverStream
   : false;

return (
   <div className="upload-overlay">
   <div className="upload-modal">
      <h2 className="upload-title">Upload Video Driver & Road</h2>
      <div className="upload-divider"></div>

      <div className="upload-boxes">
         <div className="upload-box-wrapper">
         <p className="upload-label">Video Driver</p>
         <div className="upload-box" onClick={handleDriverBoxClick}>
            {driverMode === "camera" ? (
               <>
               <video
                  ref={driverPreviewRef}
                  className="driver-preview-video"
                  autoPlay
                  muted
                  playsInline
               />
               <span className="live-badge">LIVE</span>
               </>
            ) : (
               <svg
               className="upload-icon"
               viewBox="0 0 24 24"
               fill="none"
               stroke="currentColor"
               strokeWidth="1.5"
               >
               <path d="M7 18a4.6 4.4 0 0 1-1.756-8.834A5.5 5.5 0 0 1 15.5 8.5 4.5 4.5 0 0 1 17 17H7z" />
               <path d="M12 12v6" />
               <path d="M9.5 14.5 12 12l2.5 2.5" />
               </svg>
            )}
            <input
               type="file"
               accept="video/*"
               ref={driverInputRef}
               onChange={handleDriverFileChange}
               hidden
            />
         </div>
         {driverMode === "file" && driverFile && (
            <p className="file-name">{driverFile.name}</p>
         )}
         {driverMode === "camera" && (
            <p className="file-name">Live camera active</p>
         )}
         </div>

         <span className="upload-ampersand">&amp;</span>

         <div className="upload-box-wrapper">
         <p className="upload-label">Video Road</p>
         <div className="upload-box" onClick={handleRoadBoxClick}>
            <svg
               className="upload-icon"
               viewBox="0 0 24 24"
               fill="none"
               stroke="currentColor"
               strokeWidth="1.5"
            >
               <path d="M7 18a4.6 4.4 0 0 1-1.756-8.834A5.5 5.5 0 0 1 15.5 8.5 4.5 4.5 0 0 1 17 17H7z" />
               <path d="M12 12v6" />
               <path d="M9.5 14.5 12 12l2.5 2.5" />
            </svg>
            <input
               type="file"
               accept="video/*"
               ref={roadInputRef}
               onChange={handleRoadFileChange}
               hidden
            />
         </div>
         {roadFile && <p className="file-name">{roadFile.name}</p>}
         </div>
      </div>

      <div className="upload-notes">
         <p className="notes-title">Notes:</p>
         <p>*Video Driver is video showing the driver during the trip (required).</p>
         <p>*Video Road is video showing road conditions during the trip (optional).</p>
      </div>

      <div className="upload-actions">
         <button className="btn-cancel" onClick={onClose}>
         Cancel
         </button>
         <button
         className="btn-upload"
         onClick={handleUploadClick}
         disabled={!driverReady}
         >
         Upload
         </button>
      </div>
   </div>

   {showChoice && (
      <ChoiceModal
         onChooseUpload={handleChooseUpload}
         onChooseCamera={handleChooseCamera}
         onClose={() => setShowChoice(false)}
         error={cameraError}
      />
   )}
   </div>
);
}

function ChoiceModal({ onChooseUpload, onChooseCamera, onClose, error }) {
return (
   <div className="choice-overlay" onClick={onClose}>
   <div className="choice-modal" onClick={(e) => e.stopPropagation()}>
      <h3 className="choice-title">Video Driver</h3>
      <p className="choice-question">
         Do you want to upload a video or show a live camera feed?
      </p>
      {error && <p className="camera-error">{error}</p>}
      <div className="choice-actions">
         <button className="btn-live-camera" onClick={onChooseCamera}>
         Live Camera
         </button>
         <button className="btn-upload" onClick={onChooseUpload}>
         Upload
         </button>
      </div>
   </div>
   </div>
);
}

export default UploadPage;