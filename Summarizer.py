import os
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#define function to convert speech to text
def convert_speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        print(f"Error during speech-to-text conversion: {e}")
        return None

def summarize_text(text):
    try:
        local_dir = "Range-Meeting-Summarizer/t5-base"
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(local_dir)

        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"Error in text summarization: {e}")
        return None

def main():
    audio_file = 'Range-Meeting-Summarizer/videoplayback.wav'
    
    if not os.path.isfile(audio_file):
        print(f"File '{audio_file}' not found.")
        return
    
    print("Converting speech to text...")
    meeting_text = convert_speech_to_text(audio_file)
    if meeting_text is None:
        print("Speech-to-text conversion failed.")
        return
    
    print("Transcribed Text:\n", meeting_text)
    
    print("Summarizing text...")
    summary = summarize_text(meeting_text)
    if summary is None:
        print("Text summarization failed.")
    else:
        print("Meeting Summary:\n", summary)

if __name__ == "__main__":
    main()
