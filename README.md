# Vaani — Indian Sign Language (ISL) Real-Time Translator

Vaani is an assistive computer vision application designed to help bridge communication gaps between Indian Sign Language (ISL) users and people unfamiliar with sign language. It aims to translate dynamic hand movements into text and speech in real time using computer vision and sequential landmark tracking.

## The Problem and Core Innovation

Many sign language recognition systems focus on American Sign Language (ASL) or classify isolated gestures from individual image frames. Indian Sign Language has its own vocabulary and linguistic structure, and recognizing its signs can require understanding movement over time.

Vaani focuses on two key challenges:

1. **Data scarcity:** Publicly available, standardized datasets for Indian Sign Language can be limited. Vaani provides a workflow for collecting and organizing custom regional sign-language data, allowing the dataset to be expanded over time.
2. **Temporal motion recognition:** Rather than relying exclusively on individual frames, Vaani uses a **30-frame temporal buffer** to process sequences of spatial tracking landmarks. This allows the recognition pipeline to consider hand movement and changes in position over time.

## Architecture and Features

- **Real-Time Spatial Tracking:** Uses Google MediaPipe and OpenCV to extract hand landmarks, facial landmarks, and body-pose information from a live camera feed.
- **Dynamic Action Recognition:** Processes landmark sequences through a sliding 30-frame window to support recognition of movement-based signs.
- **Custom Dataset Collection:** Includes a keypoint collection workflow for recording and labeling custom ISL signs and regional variations.
- **Sequential Model Training:** Provides a training workflow for learning patterns from collected landmark sequences.
- **Text-to-Speech Integration:** Aims to convert recognized signs into spoken output for more accessible communication.

## Technology Stack

| Technology                | Purpose                                                                    |
| ------------------------- | -------------------------------------------------------------------------- |
| Python                    | Core application logic                                                     |
| OpenCV                    | Camera input and image processing                                          |
| Google MediaPipe          | Hand, face, and pose landmark extraction                                   |
| NumPy                     | Numerical operations and landmark data processing                          |
| Scikit-learn / TensorFlow | Machine learning and sequence recognition, depending on the implementation |
| Text-to-Speech            | Spoken output for recognized signs                                         |

## Repository Structure

```text
Vaani-ISL-Translator/
├── assets/                  # UI elements and architecture diagrams
├── data/                    # Generated landmark arrays
│   └── custom_dataset/      # Labeled custom sign-language data
├── scripts/
│   ├── capture_data.py      # Collect custom landmark sequences
│   ├── train_model.py       # Train the recognition model
│   └── app_main.py          # Main real-time application
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8 or later
- A working webcam
- Git
- A compatible operating system
- The dependencies listed in `requirements.txt`

### 1. Clone the Repository

```bash
git clone https://github.com/DinoCodes-Potter/Vaani-ISL-Translator.git
cd Vaani-ISL-Translator
```

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the scripts from the repository root.

### Step 1: Collect Custom Sign Data

To record additional signs or regional variations, run:

```bash
python scripts/capture_data.py
```

The collection workflow records landmark sequences and associates them with their corresponding labels. The resulting data can be used to expand the training dataset.

### Step 2: Train the Recognition Model

After collecting the required data, run:

```bash
python scripts/train_model.py
```

This script trains or updates the recognition model using the available landmark sequences.

### Step 3: Launch the Real-Time Translator

Start the main application:

```bash
python scripts/app_main.py
```

The application initializes the camera and begins processing incoming frames for sign recognition.

> **Note:** Exact setup requirements, supported signs, model configuration, and output behavior depend on the current implementation and available training data.

## Implementation Roadmap

- [x] Develop an automated keypoint collection workflow.
- [x] Implement a 30-frame temporal tracking buffer.
- [ ] Improve multi-sign phrase stitching to combine recognized signs into grammatically meaningful sentences.
- [ ] Explore lightweight WebGL-based tracking for browser-based deployment.
- [ ] Investigate context-aware correction using language models or vision-language models.
- [ ] Expand and evaluate the dataset across regional variations, signers, and recording conditions.
- [ ] Evaluate recognition accuracy, latency, and robustness under different lighting and background conditions.

## Contributing

Contributions that improve sign recognition, accessibility, performance, and support for regional variations are welcome.

1. Fork the repository.

2. Create a feature branch:

   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. Commit your changes:

   ```bash
   git commit -m "Add AmazingFeature"
   ```

4. Push your branch:

   ```bash
   git push origin feature/AmazingFeature
   ```

5. Open a pull request describing your changes.

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

## Contact and Links

- **Developer:** Abhinav Sharma
- **GitHub:** [@DinoCodes-Potter](https://github.com/DinoCodes-Potter)
- **Project Portfolio:** Wizard Timer Live
- **Email:** [dinobusinessworks@gmail.com](mailto\:dinobusinessworks@gmail.com)

