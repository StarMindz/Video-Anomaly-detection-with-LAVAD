# LAVAD - Language-Augmented Visual Anomaly Detection

This repository contains a collection of notebooks, and experiments focused on **LAVAD** (Language-Augmented Visual Anomaly Detection) for video-based anomaly detection. The goal is to demonstrate the capabilities of generative AI models for detecting anomalies in videos. Through LAVAD, we aim to efficiently detect theft and suspicious behavior in real-time, with minimal to no training required.

### Project Overview

**LAVAD** integrates natural language processing (NLP) with computer vision (CV) models to provide enhanced anomaly detection by combining textual prompts with visual inputs. This enables the system to recognize patterns of suspicious activity, such as shoplifting or fraudulent behaviors, without needing extensive labeled datasets or model retraining.

In this repository, we experiment with various models like **BLIP-2**, **Qwen**, and **LAVAD**, along with techniques such as **FAISS** for caption refinement and similarity-based analysis. The repository also contains detailed scripts for frame extraction, caption generation, and results evaluation.

## Contents

1. **[Introduction](#introduction)**
2. **[Setup and Installation](#setup-and-installation)**
3. **[How LAVAD Works](#how-lavad-works)**
4. **[Experimentation](#experimentation-and-results)**
6. **[Contributing](#contributing)**
7. **[License](#license)**

## Introduction

LAVAD aims to improve anomaly detection by combining **Vision-Language Models (VLM)** and **Large Language Models (LLM)** to detect anomalous behaviors in a variety of environments, such as shopping malls, self-checkout stations, or public areas. The main advantage of LAVAD is its ability to perform **zero-shot anomaly detection**, where the system does not need any task-specific training data. Instead, LAVAD leverages pre-trained models and prompts to classify actions based on a combination of visual and textual information.

### Key Concepts:
- **Vision-Language Models (VLM)**: These models combine visual inputs (e.g., frames from a video) with textual descriptions to provide a deeper understanding of the visual content.
- **Large Language Models (LLM)**: LLMs like **LLAMA** **Mistral** can generate text-based anomaly scores based on input descriptions. In LAVAD, LLMs analyze the output from VLMs and provide a numerical score indicating the likelihood of anomalous behavior.

## Setup and Installation


### Requirements

1. **Python 3.8+**
2. **PyTorch 1.10+**
3. **Transformers (Hugging Face)** for pre-trained models
4. **FAISS** for efficient similarity search
5. **OpenCV** for video and image processing
6. **Sentence-Transformers** for generating text embeddings

### Installing Dependencies

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/LAVAD-Anomaly-Detection-Experiments.git
cd LAVAD-Anomaly-Detection-Experiments
run the notebooks

## How LAVAD Works

The process works as follows:
- Frame Extraction: Video is broken into individual frames using OpenCV.
- Frame Captioning: Each frame is passed to the VLM, which generates a textual description of what is happening in the frame.
- Caption Cleaning: A caption cleaning process compares each frame’s caption to those of similar frames, ensuring the description accurately matches the visual scene and removing any noise.
- Summarization: Captions across a temporal window are aggregated to create a scene summary that captures the broader context of actions over time.
- Anomaly Detection: These descriptions are then analyzed by an LLM (e.g., GPT-3) to generate an anomaly score based on predefined behavioral patterns (e.g., theft, fraud).

Large Language Models (LLM) for Anomaly Scoring
Once captions are generated for each frame, the system feeds these captions into a Large Language Model for evaluation. The LLM calculates a theft score based on the context provided in the captions, outputting a score between 0.0 (normal behavior) and 1.0 (high likelihood of theft).

LAVAD achieves zero-shot detection, meaning that it doesn't require any retraining or fine-tuning of models on labeled datasets. Instead, it uses prompt engineering to guide the model in making predictions.

## Experimentation 

Model Selection
During experimentation, different versions of the BLIP and Qwen models were evaluated for their ability to generate meaningful captions from video frames. The following models were explored:

BLIP-2: This model provides excellent frame-level captioning but requires additional fine-tuning for anomaly detection.
KOSMOS: Captions generated are more detailed but looks less accurate. The model is almost  as memory intensive as BLIP and I experimented using text prompt.
Qwen (Qwen2-VL): More powerful for handling large-scale video data, it also supports text-to-image and text-to-video tasks, making it ideal for this use case.
We also implemented FAISS for caption refinement, which helps by finding semantically similar captions and using them to refine the outputs of the model.

## 🤝 Contributing <a name="contributing"></a>

Contributions, issues, and feature requests are welcome!

Feel free to check the [issues page](../../issues/).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- SUPPORT -->

## ⭐️ Show your support <a name="support"></a>



If you like this project...

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## 📝 License <a name="license"></a>

This project is [MIT](./MIT) licensed.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



