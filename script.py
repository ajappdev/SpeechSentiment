import whisper
import os
from transformers import pipeline
import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(duration=10, samplerate=44100):
    print("\n--------------------- Recording Audio ---------------------")
    print(f"Recording for {duration} seconds. Speak now for 10 seconds...")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=2, dtype='int16')
    sd.wait()

    print("Recording complete.")
    
    # Ask the user for the output file name
    output_file = input("\nEnter the desired output file name (e.g., recording.wav): ")

    # Add .wav extension if not provided
    if not output_file.endswith(".wav"):
        output_file += ".wav"

    write(output_file, samplerate, audio_data)
    print(f"\n##### Recording saved to {output_file} #####")
    return output_file

def format_sentiment(sentiment_result):
    formatted_result = {
        'Positive': f"{sentiment_result[0]['label']}: {sentiment_result[0]['score'] * 100:.2f}%",
        'Negative': f"{sentiment_result[1]['label']}: {sentiment_result[1]['score'] * 100:.2f}%",
        'Neutral': f"{sentiment_result[2]['label']}: {sentiment_result[2]['score'] * 100:.2f}%"
    }
    return formatted_result

def sentiment_analysis(text):
    # Perform sentiment analysis
    print("\n--------------------- Sentiment Analysis: ---------------------")
    distilled_student_sentiment_classifier = pipeline(
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
        top_k=None
    )
    sentiment = distilled_student_sentiment_classifier(text)
    
    formatted_sentiment = format_sentiment(sentiment[0])
    for key, value in formatted_sentiment.items():
        print(value)

def transcrib_speech(audio_file):
    # Perform transcription
    model = whisper.load_model("base")
    audio_path = os.path.abspath(audio_file)
    result = model.transcribe(audio_path, fp16=False)
    
    print("\n--------------------- Transcription: ---------------------")
    print(result['text'])
    sentiment_analysis(result['text'])


if __name__ == "__main__":

    print("\n##############################################################################################")
    print("\n--------------------- Welcome to SpeechSentiment App! ---------------------")
    print("\n------------- Upload a speech and let me tell you how it sounds! --------------")
    print("\n##############################################################################################")

    while True:
        print("\nChoose how you want to input your speech:")
        print("1. Record your voice")
        print("2. Upload an audio file")
        print("3. Exit")
        
        choice = input("Enter the option number: ")

        if choice == "1":
            audio_file = record_audio()
        elif choice == "2":
            audio_file = input("\nEnter the absolute path of the audio file: ")
            if not os.path.isfile(audio_file):
                print("\n### Invalid file path. Please provide a valid path. ###")
                continue
        elif choice == "3":
            print("\n#############################################################################")
            print("\n--------------------- Exiting the application. Goodbye! ---------------------")
            print("\n#############################################################################")
            break
        else:
            print("\n### Invalid choice. Please enter a valid option. ###")
            continue
        
        transcrib_speech(audio_file)
        
        print("\n#############################################################################")
        print("\n--------------------- Exiting the application. Goodbye! ---------------------")
        print("\n#############################################################################")
        break
