import os
import speech_recognition as sr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def convert_speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = source.record()
        text = recognizer.recognize_google(audio_data)
    return text

def summarize_text(text):
    # Load the summarization model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

    # Encode the input text and generate the summary
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    audio_file = 'videoplayback.wav'
    
    if not os.path.isfile(audio_file):
        print(f"File '{audio_file}' not found.")
        return
    
    print("Converting speech to text...")
    try:
        meeting_text = convert_speech_to_text(audio_file)
        print("Transcribed Text:\n", meeting_text)
    except Exception as e:
        print(f"Error in speech-to-text conversion: {e}")
        return
    
    print("Summarizing text...")
    try:
        summary = summarize_text(meeting_text)
        print("Meeting Summary:\n", summary)
    except Exception as e:
        print(f"Error in text summarization: {e}")

if __name__ == "__main__":
    main()
