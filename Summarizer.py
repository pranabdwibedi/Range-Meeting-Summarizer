import os
import re
import wave
import math
from concurrent.futures import ProcessPoolExecutor
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Function to preprocess the text entered
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    filler_words = ["uh", "um", "you know", "like", "so", "basically", "actually"]
    for word in filler_words:
        text = text.replace(word, "")
    return text.strip()

# Function to split audio into smaller chunks
def split_audio(audio_file, chunk_length_ms=30000):
    with wave.open(audio_file, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()

        chunk_length_frames = int(sample_rate * chunk_length_ms / 1000)
        total_chunks = math.ceil(n_frames / chunk_length_frames)
        
        chunks = []
        for i in range(total_chunks):
            start_frame = i * chunk_length_frames
            end_frame = start_frame + chunk_length_frames
            wf.setpos(start_frame)
            frames = wf.readframes(end_frame - start_frame)
            
            chunk_filename = f"temp_chunk_{i}.wav"
            with wave.open(chunk_filename, 'wb') as chunk_wf:
                chunk_wf.setnchannels(n_channels)
                chunk_wf.setsampwidth(sample_width)
                chunk_wf.setframerate(sample_rate)
                chunk_wf.writeframes(frames)
                
            chunks.append(chunk_filename)
            
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
    finally:
        os.remove(audio_chunk)

# Function to summarize text
def summarize_text(chunk, model, tokenizer):
    try:
        inputs = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"Error in text summarization: {e}")
        return None

# Separate function to use with ProcessPoolExecutor for summarization
def summarize_chunk(args):
    chunk, model, tokenizer = args
    return summarize_text(chunk, model, tokenizer)

def main():
    audio_file = 'Meeting.wav'
    
    if not os.path.isfile(audio_file):
        print(f"File '{audio_file}' not found.")
        return
    
    print("Splitting audio into chunks...")
    audio_chunks = split_audio(audio_file)
    
    print(f"Converting speech to text for {len(audio_chunks)} chunks...")
    with ProcessPoolExecutor() as executor:
        texts = list(executor.map(convert_speech_to_text, audio_chunks))
    
    full_text = " ".join(filter(None, texts))
    
    print("Transcribed Text:\n", full_text)
    
    print("Preprocessing text...")
    preprocessed_text = preprocess_text(full_text)
    print("Preprocessed Text:\n", preprocessed_text)
    
    print("Chunking text for summarization...")
    text_chunks = [preprocessed_text[i:i + 512] for i in range(0, len(preprocessed_text), 512)]
    print(f"Text has been split into {len(text_chunks)} chunks.")
    
    print("Loading summarization model...")
    local_dir = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(local_dir)
    
    with ProcessPoolExecutor() as executor:
        summaries = list(executor.map(summarize_chunk, [(chunk, model, tokenizer) for chunk in text_chunks]))
    
    combined_summary = ' '.join(filter(None, summaries))
    print("Combined Summary:\n", combined_summary)
    
    print("Final summarizing combined summary...")
    final_summary = summarize_text(combined_summary, model, tokenizer)
    
    if final_summary is None:
        print("Final summarization failed.")
    else:
        print("Final Meeting Summary:\n", final_summary)

if __name__ == "__main__":
    main()
