# Deepfake Audio Generation and Detection (DRDO Internship Project)

This project was developed during an AIML internship at CAIR, DRDO. It focuses on two key tasks:
- Generating deepfake audio using **Tacotron 2** and **WaveGlow**
- Detecting deepfake audio using **CNN-based classifiers** on **MFCC** and **Spectrogram** features

## Highlights

- **Tacotron 2**: Used for text-to-spectrogram synthesis.
- **WaveGlow**: Converts spectrograms to realistic speech waveforms.
- **CNN Classifier**: Trained on extracted MFCCs and Mel-spectrograms for fake/real audio classification.
- Achieved **85% accuracy** with optimized feature extraction.
- Reduced **false positives** via proper data augmentation and threshold tuning.

## Directory Structure

- `generation/`: Scripts for generating deepfake audio.
- `detection/`: Scripts for extracting features and detecting fake audio.
- `models/`: Placeholder for trained models.
- `utils/`: Helper utilities.
- `data/`: Folder to store real and fake audio samples.

## Getting Started

```bash
git clone https://github.com/your-username/deepfake-audio-drdo.git
cd deepfake-audio-drdo
pip install -r requirements.txt
```

## Usage

```bash
# Generate deepfake audio
python generation/tacotron_waveglow_demo.py

# Run detection
python detection/train_cnn_classifier.py
python detection/evaluate_classifier.py
```
        