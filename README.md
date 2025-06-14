<p align="center">
  <img src="Media\images\Fully_Logo_White.png" alt="Natq Logo" width="800"/>
</p>

<h1 align="center">🗣️ Natq – Arabic Text-to-Speech (TTS) System</h1>

<p align="center">

## 📌 Overview

Natq is an open-source Arabic Text-to-Speech (TTS) system that uses deep learning models to convert Arabic text into natural-sounding speech.  
We designed and trained multiple architectures on public datasets with a focus on Arabic language intricacies, including diacritization and phoneme modeling.

---
<p align="center">
<img src="Media\images\Helwan_University_Logo.jpg" alt="Natq Logo" width="300"/>
</p>
<p align="center">
<b>Graduation Project – Faculty of Computing and Artificial Intelligence, Helwan University<br>
Department of Artificial Intelligence<br>
Supervised by: Dr. Yasser Hifny & Dr. Ahmed Hesham
</b>
  </p>

## 🎯 Objectives

- Create a modular and efficient Arabic TTS system using public datasets.
- Handle Arabic-specific challenges such as diacritics and phoneme mapping.
- Compare performance of different spectrogram generators and vocoders.
- Support live speech synthesis via a web app (React + FastAPI).

---

## 🧠 Architecture

The system pipeline is composed of:

1. **Text Preprocessing**
2. **Diacritization**: [CATT Transformer](https://arxiv.org/abs/2306.07076)
3. **Grapheme-to-Phoneme (G2P)**: Nawar Halabi’s Phenomizer
4. **Spectrogram Generation**: FastPitch, FastSpeech2, Mixer-TTS, or Spark-TTS
5. **Vocoder**: HiFi-GAN for waveform synthesis

> ![Pipeline Diagram](images/tts_pipeline.png)

---

## 📚 Datasets

| Dataset   | Hours | Diacritized | Accent        | Notes                       |
|-----------|-------|-------------|---------------|-----------------------------|
| Arabic_Speech_Corpus    | 4.1   | ✅          | Levantine     | Open-source                 |
| ClArTTS   | 12    | ✅          | Classical MSA | Based on LibriVox audiobook |

---

## 🛠 Models Used

### 🔹 Spectrogram Generators

- **FastPitch** – Parallel and pitch-controllable TTS
- **FastSpeech2** – Variance-aware and efficient
- **Mixer-TTS** – MLP-Mixer based parallel synthesis
- **Spark-TTS** – End-to-end LLM-based TTS with zero-shot speaker cloning

### 🔹 Vocoder

- **HiFi-GAN** – Fast, high-fidelity waveform generation

> ![Spectrogram Comparison](images/spectrogram_comparison.png)

---

## 🧪 Evaluation

### Objective

- Spectrogram quality
- Inference time
- Model size

### Subjective

- Mean Opinion Score (MOS) from human listeners

> ![MOS Table](images/mos_table.png)

---

<!-- ## 🔈 Audio Samples

| Model      | Dataset  | Sample             |
|------------|----------|--------------------|
| FastPitch  | ClArTTS  | 🔊 [Listen](audio_samples/fastpitch_sample.wav)  
| Mixer-TTS  | ASC      | 🔊 [Listen](audio_samples/mixer_tts_sample.wav)  
| Spark-TTS  | ClArTTS  | 🔊 [Listen](audio_samples/spark_tts_sample.wav)  
| HiFi-GAN   | —        | 🔊 [Listen](audio_samples/hifigan_sample.wav) -->

## 🔈 Audio Samples

<h1 align="center">وَالسَّلَامُ عَلَى أَشْرَفِ الْأَنْبِيَاءِ وَالْمُرْسَلِينَ سَيِّدِنَا مُحَمَّدٍ</h1>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Dataset</th>
        <th>Sample</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>FastPitch</td>
        <td>ClArTTS</td>
        <td>
          <audio controls>
            <source src="Media/audio_samples/FastPitch-TTS_MOS/walsalam_fp.wav" type="audio/wav">
            Your browser does not support the audio element.
          </audio>
        </td>
      </tr>
      <tr>
        <td>Mixer-TTS</td>
        <td>ASC</td>
        <td>
          <audio controls>
            <source src="Media/audio_samples/Mixer-TTS_MOS/walsalam_mx.wav" type="audio/wav">
            Your browser does not support the audio element.
          </audio>
        </td>
      </tr>
      <tr>
        <td>Spark-TTS</td>
        <td>ClArTTS</td>
        <td>
          <audio controls>
            <source src="Media/audio_samples/Spark-TTS_MOS/salam_spark.wav" type="audio/wav">
            Your browser does not support the audio element.
          </audio>
        </td>
      </tr>
      <tr>
        <td>VITS_facebook</td>
        <td>—</td>
        <td>
          <audio controls>
            <source src="Media/audio_samples/speech_VITS_facebook_output.wav" type="audio/wav">
            Your browser does not support the audio element.
          </audio>
        </td>
      </tr>
      <tr>
        <td>T5_MBZUAI</td>
        <td>—</td>
        <td>
          <audio controls>
            <source src="Media/audio_samples/speech_T5_MBZUAI_output.wav" type="audio/wav">
            Your browser does not support the audio element.
          </audio>
        </td>
      </tr>
      <tr>
        <td>FastSpeech2</td>
        <td>—</td>
        <td>
          <audio controls>
            <source src="audio_samples/hifigan_sample.wav" type="audio/wav">
            Your browser does not support the audio element.
          </audio>
        </td>
      </tr>
    </tbody>
  </table>

---

## 👥 Contributors

- [Abdelhalim Ashraf](https://github.com/ABDELHALIM9)
- [Abdelrahman Ramadan](https://github.com/Abdelrhman-Ramadan)
- [Eid Osama Eid](https://github.com/eid-osama)
- [Omar Tamer](https://github.com/Omartamer783)
- [Ali Adel](http://github.com/ali-adel)

---

## 📂 Project Structure

```graphql
d:/coding/Natq/
├── .gitignore
├── FastPitch/
│   ├── .vscode/
│   │   └── sttings.json
│   ├── arabic_phoneme_tokenizer.py
│   ├── catt/
│   ├── custom_arabic_to_phones.py
│   └── ...
├── FastSpeech2/
│   ├── arabic_phoneme_tokenizer.py
│   ├── catt/
│   └── ...
├── Media/
│   ├── audio_samples/
│   │   ├── FastPitch-TTS_MOS/
│   │   │   ├── bism_fp.mp4
│   │   │   ├── bism_fp.wav
│   │   │   └── ...
│   │   ├── Mixer-TTS_MOS/
│   │   │   ├── bism_mix.mp4
│   │   │   ├── bism_mix.wav
│   │   │   └── ...
│   │   └── Spark-TTS_MOS/
│   │       ├── bism_spark.mp4
│   │       ├── bism_spark.wav
│   │       └── ...
│   └── images/
│       ├── Demo.jpg
│       └── Fully_Logo_White.png
├── Natq-Frontend/
│   ├── .gitignore
│   ├── eslint.config.js
│   └── ...
├── README.md
└── Spark/
    ├── catt/
    ├── download_model_files.ipynb
    └── ...

```

---

## 🌐 Web Application

We built a simple TTS web interface using **FastAPI** for the backend and **React** for the frontend.

<p align="center">
  <img src="Media\images\Demo.jpg" alt="Natq Logo" width="800"/>
</p>

🧪 Test it locally:

```bash
cd backend
uvicorn main:app --reload
```

# Then in another terminal

```bash
cd ../app
npm install
npm start
```
