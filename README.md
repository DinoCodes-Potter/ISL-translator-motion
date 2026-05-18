# Vaani — Indian Sign Language (ISL) Real-Time Translator

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Computer Vision](https://img.shields.io/badge/Computer--Vision-MediaPipe%20%7C%20OpenCV-darkgreen.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Accessibility](https://img.shields.io/badge/Accessibility-Automated%20Infrastructure-orange.svg)]()

Vaani is an assistive computer vision application designed to bridge communication gaps for the deaf and hard-of-hearing community in India by translating fluid Indian Sign Language (ISL) into real-time text and speech[span_0](start_span)[span_0](end_span).

---

## The Problem & The Core Innovation

Most public sign language translation models are optimized for American Sign Language (ASL) and rely heavily on static hand gestures (interpreting a single snapshot frame). 

Indian Sign Language (ISL) is highly structural, contextual, and dynamic. It relies heavily on spatial movement, facial grammar, and fluid transitions. Vaani addresses two fundamental barriers:

1. **The Data Scarcity Problem:** There is a severe lack of high-fidelity, standardized, public datasets for ISL[span_1](start_span)[span_1](end_span). Vaani overcomes this by incorporating a pipeline built on custom, systematically curated regional dataset buffers[span_2](start_span)[span_2](end_span).
2. **The Temporal Motion Challenge:** Instead of evaluating static isolated frames, Vaani implements a custom **30-frame temporal buffer system**[span_3](start_span)[span_3](end_span). This architecture processes sequence-based spatial tracking vectors over time, allowing it to correctly translate fluid hand motion, gestural direction, and kinetic sign language phrases[span_4](start_span)[span_4](end_span).

---

## Architecture & Core Features

* **Real-time Spatial Tracking:** Utilizes Google MediaPipe's Holistic pipeline to extract real-time topographies of hands (21 landmarks per hand), facial landmarks, and pose structures without heavy GPU hardware dependency.
* **Dynamic Action Recognition:** Processes landmark sequences over a sliding 30-frame window to interpret context and motion-based vocabulary[span_5](start_span)[span_5](end_span).
* **Custom Dataset Augmentation Matrix:** Features a built-in automated keypoint collection script allowing developers to quickly record, process, and append custom regional ISL phrases locally[span_6](start_span)[span_6](end_span).
* **Text-to-Speech Integration:** Outputs parsed words and continuous phrases directly to audio output layers for seamless real-time dialogue.

---

## Technology Stack

* **Core Logic:** Python (Advanced logical processing)[span_7](start_span)[span_7](end_span)
* **Computer Vision Frameworks:** OpenCV, Google MediaPipe Ecosystem[span_8](start_span)[span_8](end_span)
* **Mathematical Array Processing:** NumPy[span_9](start_span)[span_9](end_span)
* **Machine Learning Layer:** Scikit-Learn / TensorFlow (Sequential Landmark Pipelines)[span_10](start_span)[span_10](end_span)

---

## Repository Structure

```text
├── assets/                  # UI Elements and architecture diagrams
├── data/                    # Generated numpy arrays of tracking points
│   └── custom_dataset/      # Structured directories of labeled keypoint landmarks
├── scripts/
│   ├── capture_data.py      # Script to systematically generate custom tracking data
│   ├── train_model.py       # Sequence processing & model fitting script
│   └── app_main.py          # Main real-time application runtime loop
├── requirements.txt         # Project runtime dependencies
└── README.md
Getting Started & Installation
Prerequisites
Ensure you have Python 3.8+ and a working web camera connected to your system.
1. Clone the Repository
git clone [https://github.com/DinoCodes-Potter/Vaani-ISL-Translator.git](https://github.com/DinoCodes-Potter/Vaani-ISL-Translator.git)
cd Vaani-ISL-Translator
2. Set Up a Virtual Environment (Recommended)
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
Technical Execution & Usage
Step 1: Custom Keypoint Collection (capture_data.py)
If you wish to augment the training sequence with custom phrases or regional signs, launch the capture engine. It records sequence paths across 30 distinct intervals:
python scripts/capture_data.py
The script initializes your video capture layout, generates specialized labeling arrays, and creates sequential .npy records of spatial hand structural frames directly into the data/ matrix.
Step 2: Sequence Engine Training (train_model.py)
Train or update the temporal parsing logic using your recorded sequence array map:
python scripts/train_model.py
Step 3: Run Main Real-Time Translation Application
Launch the live webcam interface layer to begin real-time translation:
python scripts/app_main.py
Implementation Roadmap
• [x] Create automated keypoint generation workflow scripts.  
• [x] Deploy 30-frame multi-temporal tracking buffer algorithms.  
• [ ] Implement multi-phrase logic stitching (combining single words into structural syntax sentences).
• [ ] Port spatial tracking parameters onto lightweight WebGL frameworks to achieve zero-install web capability.
• [ ] Introduce custom fine-tuning modules leveraging advanced LLM/VLM APIs for deep context correction.
Contributing
Contributions to help bridge communication gaps across India are welcome!
1.	Fork the Repository.
2.	Create a feature branch (git checkout -b feature/AmazingFeature).
3.	Commit your changes (git commit -m 'Add some AmazingFeature').
4.	Push to the branch (git push origin feature/AmazingFeature).
5.	Open a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Contact & Links
• Developer: Abhinav Sharma  
• GitHub Profile: @DinoCodes-Potter  
• Project Portfolio: Wizard Timer Live  
• Email: dinobusinessworks@gmail.com
