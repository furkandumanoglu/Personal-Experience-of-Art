import { useEffect, useRef, useState } from 'react';
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import './App.css';

const App = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [poseLandmarker, setPoseLandmarker] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const requestRef = useRef();

    // Initialize MediaPipe Pose Landmarker
    useEffect(() => {
        const initPose = async () => {
            try {
                const vision = await FilesetResolver.forVisionTasks(
                    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
                );
                const landmarker = await PoseLandmarker.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task`,
                        delegate: "GPU"
                    },
                    runningMode: "VIDEO",
                    numPoses: 1
                });
                setPoseLandmarker(landmarker);
                setIsLoading(false);
            } catch (err) {
                console.error("Error initializing MediaPipe:", err);
                setError("Failed to load Pose Detection model.");
                setIsLoading(false);
            }
        };
        initPose();

        return () => {
            if (poseLandmarker) {
                poseLandmarker.close();
            }
        };
    }, []);

    // Setup Webcam
    useEffect(() => {
        const setupCamera = async () => {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                setError("Webcam access not supported in this browser.");
                return;
            }

            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480 },
                    audio: false,
                });
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    videoRef.current.onloadedmetadata = () => {
                        videoRef.current.play();
                    };
                }
            } catch (err) {
                console.error("Error accessing webcam:", err);
                setError("Camera permission denied or camera not found.");
            }
        };
        setupCamera();
    }, []);

    // Rendering Loop
    const drawPose = () => {
        if (!poseLandmarker || !videoRef.current || !canvasRef.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        if (video.readyState >= 2) { // HAVE_CURRENT_DATA
            const startTimeMs = performance.now();
            const results = poseLandmarker.detectForVideo(video, startTimeMs);

            // Clear Canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Mirror horizontally to match mirrored video
            ctx.save();
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);

            if (results.landmarks && results.landmarks.length > 0) {
                for (const landmarks of results.landmarks) {
                    // Draw connections
                    PoseLandmarker.POSE_CONNECTIONS.forEach(([startIdx, endIdx]) => {
                        const start = landmarks[startIdx];
                        const end = landmarks[endIdx];
                        if (start && end && start.visibility > 0.5 && end.visibility > 0.5) {
                            ctx.beginPath();
                            ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
                            ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
                            ctx.strokeStyle = '#00ffcc';
                            ctx.lineWidth = 3;
                            ctx.stroke();
                        }
                    });

                    // Draw keypoints
                    landmarks.forEach((landmark, index) => {
                        if (landmark.visibility > 0.5) {
                            // Only draw major keypoints ( shoulders, elbows, wrists, hips, knees, ankles)
                            // Indices: 11-16 (arms), 23-28 (legs)
                            if (index >= 11 && index <= 28) {
                                ctx.beginPath();
                                ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 5, 0, 2 * Math.PI);
                                ctx.fillStyle = '#ffffff';
                                ctx.fill();
                                ctx.strokeStyle = '#000000';
                                ctx.lineWidth = 1;
                                ctx.stroke();
                            }
                        }
                    });
                }
            }
            ctx.restore();
        }

        requestRef.current = requestAnimationFrame(drawPose);
    };

    useEffect(() => {
        if (poseLandmarker) {
            requestRef.current = requestAnimationFrame(drawPose);
        }
        return () => cancelAnimationFrame(requestRef.current);
    }, [poseLandmarker]);

    return (
        <div className="App">
            <h1 className="title">Personal Art Experience</h1>

            <div className="container">
                <video
                    ref={videoRef}
                    className="video-feed"
                    playsInline
                    muted
                    autoPlay
                />
                <canvas
                    ref={canvasRef}
                    width={640}
                    height={480}
                    className="canvas-overlay"
                />

                {(isLoading || !poseLandmarker) && (
                    <div className="status-overlay">
                        <div className="status-text">{error || "Initializing Experience..."}</div>
                    </div>
                )}
            </div>

            <div className="controls">
                <button onClick={() => window.location.reload()}>Reset System</button>
            </div>
        </div>
    );
};

export default App;
