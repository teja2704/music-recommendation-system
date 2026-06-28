const cameraPreview = document.querySelector("#camera-preview");
const captureCanvas = document.querySelector("#capture-canvas");
const startCameraButton = document.querySelector("#start-camera");
const captureFrameButton = document.querySelector("#capture-frame");
const loadSongsButton = document.querySelector("#load-songs");
const message = document.querySelector("#message");
const resultCard = document.querySelector("#result-card");
const songsCard = document.querySelector("#songs-card");
const songsList = document.querySelector("#songs-list");
const emotionLabel = document.querySelector("#emotion-label");
const emotionConfidence = document.querySelector("#emotion-confidence");
const modelStatus = document.querySelector("#model-status");
const spotifyStatus = document.querySelector("#spotify-status");

let currentEmotion = null;

function setMessage(text, isError = false) {
    message.textContent = text;
    message.classList.toggle("error-text", isError);
}

async function loadStatus() {
    const response = await fetch("/api/status");
    const status = await response.json();

    modelStatus.textContent = status.model_ready ? "Ready" : "Missing";
    spotifyStatus.textContent = status.spotify_ready ? "Ready" : "Needs configuration";

    if (status.model_error) {
        setMessage(status.model_error, true);
    }
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
        });
        cameraPreview.srcObject = stream;
        captureFrameButton.disabled = false;
        setMessage("Camera is ready. Capture when your face is visible.");
    } catch (error) {
        setMessage(`Unable to start camera: ${error.message}`, true);
    }
}

async function captureEmotion() {
    const width = cameraPreview.videoWidth;
    const height = cameraPreview.videoHeight;
    if (!width || !height) {
        setMessage("Camera stream is not ready yet.", true);
        return;
    }

    captureCanvas.width = width;
    captureCanvas.height = height;
    const context = captureCanvas.getContext("2d");
    context.drawImage(cameraPreview, 0, 0, width, height);

    setMessage("Sending captured frame for prediction...");
    const image = captureCanvas.toDataURL("image/jpeg", 0.9);
    const response = await fetch("/api/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({image}),
    });
    const result = await response.json();

    if (!response.ok) {
        setMessage(result.error || "Prediction failed.", true);
        return;
    }

    currentEmotion = result.emotion;
    emotionLabel.textContent = result.emotion;
    emotionConfidence.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
    resultCard.hidden = false;
    songsCard.hidden = true;
    songsList.innerHTML = "";
    setMessage("Emotion captured. You can now load song recommendations.");
}

function renderSongs(songs) {
    songsList.innerHTML = "";
    for (const song of songs) {
        const article = document.createElement("article");
        article.className = "song";

        if (song.album_art) {
            const albumArt = document.createElement("img");
            albumArt.className = "album-art";
            albumArt.src = song.album_art;
            albumArt.alt = "";
            article.appendChild(albumArt);
        }

        const songInfo = document.createElement("div");
        songInfo.className = "song-info";

        const title = document.createElement("h3");
        title.textContent = song.name;

        const artist = document.createElement("p");
        artist.textContent = song.artist;

        const releaseDate = document.createElement("p");
        releaseDate.className = "muted";
        releaseDate.textContent = `Released: ${song.release_date}`;

        const link = document.createElement("a");
        link.className = "play-button";
        link.href = song.url;
        link.target = "_blank";
        link.rel = "noreferrer";
        link.textContent = "Play on Spotify";

        songInfo.append(title, artist, releaseDate, link);
        article.appendChild(songInfo);
        songsList.appendChild(article);
    }
}

async function loadSongs() {
    if (!currentEmotion) {
        setMessage("Capture an emotion before loading songs.", true);
        return;
    }

    setMessage("Loading Spotify recommendations...");
    const response = await fetch(`/api/recommendations/${currentEmotion}`);
    const result = await response.json();

    if (!response.ok) {
        setMessage(result.error || "Could not load recommendations.", true);
        return;
    }

    renderSongs(result.songs);
    songsCard.hidden = false;
    setMessage("Recommendations loaded.");
}

startCameraButton.addEventListener("click", startCamera);
captureFrameButton.addEventListener("click", captureEmotion);
loadSongsButton.addEventListener("click", loadSongs);
loadStatus().catch((error) => setMessage(error.message, true));
