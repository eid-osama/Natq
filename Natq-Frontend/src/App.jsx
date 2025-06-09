import { useState } from "react";
import "tailwindcss";

function App() {
  const [text, setText] = useState("");
  const [model, setModel] = useState("fastpitch");
  const [audioUrl, setAudioUrl] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setAudioUrl(null);
    setLoading(true);

    let endpoint = "";
    if (model === "fastpitch") {
      endpoint = "http://localhost:8001/synthesize/fastpitch";
    } else if (model === "fastspeech2") {
      endpoint = "http://localhost:8000/synthesize/fastspeech2";
    }

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model }),
      });

      if (response.ok) {
        const blob = await response.blob();
        setAudioUrl(URL.createObjectURL(blob));
      } else {
        alert("Error generating audio");
      }
    } catch (err) {
      console.error(err);
      alert("Failed to connect to the server.");
    }
    setLoading(false);
  };

  return (
    <>
      <nav></nav>
      <div className="bg-[#1C1B43] text-white flex place-content-center">
        <div className="w-[1300px] h-[640px] flex place-content-center py-[22px] my-[140px] conic-gradient rounded-[20px]">
          <div className="max-w-[625px] max-h-[596px] w-full">
            <img
              className="max-w-[625px] max-h-[596px] w-full"
              src="/images/Speechbot.svg"
              alt=""
            />
          </div>
          <div className="max-w-[625px] max-h-[596px] w-full bg-[#1C1B43] rounded-tr-[20px] rounded-br-[20px]  px-[30px]">
            <div className="flex justify-between pt-[14px]">
              <span className="place-content-center">speechbot</span>
              <img src="/images/waveform-icon.svg" alt="" />
            </div>
            <form className="max-h-[500px] h-full" onSubmit={handleSubmit}>
              <textarea
                dir="rtl"
                placeholder="ادخل النص"
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="resize-none focus:outline-none bg-[#1E2A52] border-4 border-[#27456D] rounded-[20px] max-w-[560px] w-full h-[170px] mt-[38px] mb-[54px] p-[30px]"
              ></textarea>
              <select
                className="focus:outline-none bg-[#1E2A52] border-4 border-[#27456D] rounded-[20px] max-w-[560px] w-full h-[60px] mb-[20px] px-[20px]"
                name="model"
                id="model"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              >
                <option value="fastpitch">FastPitch</option>
                <option value="fastspeech2">FastSpeech2</option>
              </select>
              <div className="h-[80px] mb-[20px] pt-[15px]">
                {audioUrl && <audio controls src={audioUrl} />}
              </div>
              <button
                type="submit"
                className="btn-gradient max-w-[560px] w-full h-[50px] rounded-[15px] cursor-pointer"
                disabled={loading}
              >
                {loading ? "Loading.." : "convert⚡"}
              </button>
            </form>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
