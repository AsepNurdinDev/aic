import { useState, useRef, useEffect } from "react";
import UploadPage from "./UploadPage";
import "../App.css";

function VideoCard({ label, source, videoRef, onPause, onEnded }) {
useEffect(() => {
   if (source?.type === "camera" && videoRef.current) {
   videoRef.current.srcObject = source.stream;
   }
}, [source, videoRef]);

return (
   <div className="video-card">
   {source?.type === "file" && (
      <video
         ref={videoRef}
         src={source.url}
         className="video-player"
         controls
         onPause={onPause}
         onEnded={onEnded}
      />
   )}

   {source?.type === "camera" && (
      <>
         <video
         ref={videoRef}
         className="video-player"
         autoPlay
         muted
         playsInline
         />
         <span className="live-badge">LIVE</span>
      </>
   )}

   {!source && <h2>{label}</h2>}
   </div>
);
}

function HomePage() {
const [showUpload, setShowUpload] = useState(false);
const [driverSource, setDriverSource] = useState(null); // {type:'file',url} | {type:'camera',stream} | null
const [roadSource, setRoadSource] = useState(null); // {type:'file',url} | null
const [isPlaying, setIsPlaying] = useState(false);

const driverRef = useRef(null);
const roadRef = useRef(null);

const handleUpload = ({ driver, road }) => {
   if (driver?.type === "file") {
   setDriverSource({ type: "file", url: URL.createObjectURL(driver.file) });
   } else if (driver?.type === "camera") {
   setDriverSource({ type: "camera", stream: driver.stream });
   }

   if (road?.type === "file") {
   setRoadSource({ type: "file", url: URL.createObjectURL(road.file) });
   }

   setShowUpload(false);
};

// Matikan kamera hanya kalau HomePage-nya sendiri unmount (misal pindah halaman)
useEffect(() => {
   return () => {
   if (driverSource?.type === "camera") {
      driverSource.stream.getTracks().forEach((t) => t.stop());
   }
   };
}, [driverSource]);

// Tombol tengah hanya mengontrol video berbasis FILE (road, dan driver kalau dia file juga).
// Live camera driver otomatis selalu berjalan (real-time), jadi tidak perlu di-play/pause.
const handleTogglePlay = () => {
   const playableRefs = [];
   if (driverSource?.type === "file" && driverRef.current) playableRefs.push(driverRef.current);
   if (roadSource?.type === "file" && roadRef.current) playableRefs.push(roadRef.current);

   if (playableRefs.length === 0) return;

   if (isPlaying) {
   playableRefs.forEach((v) => v.pause());
   setIsPlaying(false);
   } else {
   playableRefs.forEach((v) => {
      v.currentTime = 0;
      v.play();
   });
   setIsPlaying(true);
   }
};

const handleAnyPause = () => setIsPlaying(false);

const bothReady = driverSource && roadSource;
const hasPlayableFile =
   driverSource?.type === "file" || roadSource?.type === "file";

return (
   <main className="home">
   <header className="header">
      <h1>AIC — SafeRoute AI</h1>
      <button onClick={() => setShowUpload(true)}>UPLOAD</button>
   </header>

   <section className="video-section">
      <VideoCard
         label="driver video"
         source={driverSource}
         videoRef={driverRef}
         onPause={handleAnyPause}
         onEnded={handleAnyPause}
      />

      {bothReady && hasPlayableFile && (
         <button
         className="play-both-button"
         onClick={handleTogglePlay}
         aria-label={isPlaying ? "Pause videos" : "Play videos"}
         >
         {isPlaying ? (
            <svg viewBox="0 0 24 24" fill="currentColor">
               <rect x="6" y="5" width="4" height="14" />
               <rect x="14" y="5" width="4" height="14" />
            </svg>
         ) : (
            <svg viewBox="0 0 24 24" fill="currentColor">
               <path d="M8 5v14l11-7z" />
            </svg>
         )}
         </button>
      )}

      <VideoCard
         label="road video"
         source={roadSource}
         videoRef={roadRef}
         onPause={handleAnyPause}
         onEnded={handleAnyPause}
      />
   </section>

   <section className="condition">
      <h2>Current Condition</h2>
   </section>

   {showUpload && (
      <UploadPage
         onClose={() => setShowUpload(false)}
         onUpload={handleUpload}
      />
   )}
   </main>
);
}

export default HomePage;