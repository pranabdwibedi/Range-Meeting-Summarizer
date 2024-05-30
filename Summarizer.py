import os
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from pydub import AudioSegment

# Function to preprocess the text entered
def preprocess_text(text):
    # Remove unwanted characters and extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove filler words and irrelevant phrases (example)
    filler_words = ["uh", "um", "you know", "like", "so", "basically", "actually"]
    for word in filler_words:
        text = text.replace(word, "")
    return text

# Function to split audio into smaller chunks
def split_audio(audio_file, chunk_length_ms=60000):
    audio = AudioSegment.from_wav(audio_file)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunks.append(chunk)
    return chunks

# Function to convert speech to text
def convert_speech_to_text(audio_chunk):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_chunk) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        print(f"Error during speech-to-text conversion: {e}")
        return None

# Function to summarize text
def summarize_text(text, model, tokenizer):
    try:
        # Encode the text
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        # Generate the summary
        summary_ids = model.generate(
            inputs, 
            max_length=150, 
            min_length=50, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"Error in text summarization: {e}")
        return None

def main():
    audio_file = 'Project1/Meeting.wav'
    
    if not os.path.isfile(audio_file):
        print(f"File '{audio_file}' not found.")
        return
    
    print("Splitting audio into chunks...")
    audio_chunks = split_audio(audio_file)
    
    print(f"Converting speech to text for {len(audio_chunks)} chunks...")
    full_text = ""
    for i, chunk in enumerate(audio_chunks):
        chunk.export("temp_chunk.wav", format="wav")
        chunk_text = convert_speech_to_text("temp_chunk.wav")
        if chunk_text is None:
            print(f"Speech-to-text conversion failed for chunk {i + 1}.")
            return
        full_text += chunk_text + " "
    
    print("Transcribed Text:\n", full_text)
    
    print("Preprocessing text...")
    preprocessed_text = preprocess_text(full_text)
    print("Preprocessed Text:\n", preprocessed_text)
    
    print("Chunking text for summarization...")
    text_chunks = chunk_text(preprocessed_text)
    print(f"Text has been split into {len(text_chunks)} chunks.")
    
    print("Loading summarization model...")
    local_dir = "Project1/t5-base"
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(local_dir)
    
    summaries = []
    for i, chunk in enumerate(text_chunks):
        print(f"Summarizing text chunk {i + 1}/{len(text_chunks)}...")
        summary = summarize_text(chunk, model, tokenizer)
        if summary:
            summaries.append(summary)
    
    combined_summary = ' '.join(summaries)
    print("Combined Summary:\n", combined_summary)
    
    print("Final summarizing combined summary...")
    final_summary = summarize_text(combined_summary, model, tokenizer)
    
    if final_summary is None:
        print("Final summarization failed.")
    else:
        print("Final Meeting Summary:\n", final_summary)

if __name__ == "__main__":
    main()
