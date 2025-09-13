import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [inputUrl, setInputUrl] = useState(null);
  const [maskUrl, setMaskUrl] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setInputUrl(null);
    setMaskUrl(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      // URLs for input and predicted mask images
      setInputUrl(`http://127.0.0.1:8000/${data.input}`);
      setMaskUrl(`http://127.0.0.1:8000/${data.mask}`);
    } catch (error) {
      console.error("Error uploading image:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Flood Segmentation Demo</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={!file || loading}>
        {loading ? "Processing..." : "Upload & Predict"}
      </button>

      {maskUrl && inputUrl && (
        <div
          style={{
            display: "flex",
            justifyContent: "center", // centers images horizontally
            alignItems: "center",
            gap: "20px",
            marginTop: "20px",
          }}
        >
          <div style={{ textAlign: "center" }}>
            <h2>Preprocessed Input:</h2>
            <img
              src={`${inputUrl}?t=${Date.now()}`}
              alt="Input Image"
              style={{ maxWidth: "300px", height: "auto" }}
            />
          </div>
          <div style={{ textAlign: "center" }}>
            <h2>Predicted Mask:</h2>
            <img
              src={`${maskUrl}?t=${Date.now()}`}
              alt="Predicted Mask"
              style={{ maxWidth: "300px", height: "auto" }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;