# SpeechSentiment App

SpeechSentiment is a command line Python application that allows users to analyze the sentiment of recorded speech or uploaded audio files. It utilizes the Whisper model from OpenAI for transcription and the lxyuan/distilbert-base-multilingual-cased-sentiments-student model from Hugging Face for sentiment analysis.

## Technologies Used

- Python
- [Whisper model from OpenAI](https://whisper-api.readthedocs.io/)
- [DistilBERT-based Sentiment Analysis model from Hugging Face](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student)
- **sounddevice:** Library for recording and playing audio.
- **scipy:** Library for handling audio file I/O.


## How to Use

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ajappdev/SpeechSentiment.git```
2. Navigate to the project directory:

   ```bash
   cd SpeechSentiment
   ```
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Run the application:

   ```bash
   python script.py```
   
Choose option 1 to record your voice. Follow the on-screen instructions to record for 10 seconds. The recorded audio will be saved in the root folder. Or Choose option 2 and enter the absolute path of the audio file when prompted. Ensure the file exists at the specified path.

### Test Audios

The **'test audios'** folder contains sample audio files for testing. 

