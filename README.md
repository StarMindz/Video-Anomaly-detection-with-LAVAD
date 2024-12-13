# LAVAD - Language-Augmented Visual Anomaly Detection

This repository contains a collection of notebooks, and experiments focused on **LAVAD** (Language-Augmented Visual Anomaly Detection) for video-based anomaly detection. The goal is to demonstrate the capabilities of generative AI models for detecting anomalies in videos. Through LAVAD, we aim to efficiently detect theft and suspicious behavior in real-time, with minimal to no training required.

### Project Overview

**LAVAD** integrates natural language processing (NLP) with computer vision (CV) models to provide enhanced anomaly detection by combining textual prompts with visual inputs. This enables the system to recognize patterns of suspicious activity, such as shoplifting or fraudulent behaviors, without needing extensive labeled datasets or model retraining.

In this repository, we experiment with various models like **BLIP-2**, **Qwen**, and **LAVAD**, along with techniques such as **FAISS** for caption refinement and similarity-based analysis. The repository also contains detailed scripts for frame extraction, caption generation, and results evaluation.

## Contents

1. **[Introduction](#introduction)**
2. **[Setup and Installation](#setup-and-installation)**
3. **[How LAVAD Works](#how-lavad-works)**
4. **[Experimentation and Results](#experimentation-and-results)**
5. **[Usage](#usage)**
6. **[Folder Structure](#folder-structure)**
7. **[Scripts](#scripts)**
8. **[Notebooks](#notebooks)**
9. **[Results](#results)**
10. **[Contributing](#contributing)**
11. **[License](#license)**

## Introduction

LAVAD aims to improve anomaly detection by combining **Vision-Language Models (VLM)** and **Large Language Models (LLM)** to detect anomalous behaviors in a variety of environments, such as shopping malls, self-checkout stations, or public areas. The main advantage of LAVAD is its ability to perform **zero-shot anomaly detection**, where the system does not need any task-specific training data. Instead, LAVAD leverages pre-trained models and prompts to classify actions based on a combination of visual and textual information.

### Key Concepts:
- **Vision-Language Models (VLM)**: These models combine visual inputs (e.g., frames from a video) with textual descriptions to provide a deeper understanding of the visual content.
- **Large Language Models (LLM)**: LLMs like **GPT-3** or **T5** can generate text-based anomaly scores based on input descriptions. In LAVAD, LLMs analyze the output from VLMs and provide a numerical score indicating the likelihood of anomalous behavior.

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
pip install -r requirements.txt
