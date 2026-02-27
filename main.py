import os
from dotenv import load_dotenv
import soundfile as sf
import numpy as np
import torch
from EmotionDetection.models import EmotionDetector
from Qwen3_tts.models.models import VoiceGenerator

load_dotenv()

model_path = os.getenv("MODEL_PATH")
# tokenizer_path = os.getenv("TOKENIZER_PATH")

text = """
    The sun was shining brightly over the meadow. Birds were singing their cheerful songs.
    Suddenly, dark clouds gathered in the sky. A loud thunder echoed across the valley.
    The little girl felt afraid and ran towards her home. But then she remembered her mother's words.
    She took a deep breath and smiled. Everything would be okay.
    She danced in the rain, feeling free and happy. What a wonderful day it turned out to be!
    """


def audio_coverter_torch_to_numpy(audio):
    '''
    This function convert the audio file which is in the Torch.Tensor to numpy array .
    Since soundfile works with numpy array 
    '''
    if isinstance(audio, torch.Tensor):
        audio = audio.squeeze().detach().cpu().numpy()
    elif isinstance(audio, list):
        audio = np.array(audio)
        
    # Ensure it's float32 (standard for wav)
    if isinstance(audio, np.ndarray):
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # Fix shape if it's (1, N) -> (N,)
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio.squeeze()
    return audio

emotion_detector = EmotionDetector(text)
emotions = emotion_detector.main()
voice_generate = VoiceGenerator(text=text,emotions=emotions,model_path=model_path)
voice_generator = voice_generate.main()


# Create an output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

#Instead of making separate files and merging them, we can directly merge them to the numpy array itself
all_audio =[]
sample_rate=None 

for i in range(len(voice_generator)):
    audio_data, sr = voice_generator[i]
    if sample_rate is None:
        sample_rate = int(sr)
    audio = audio_coverter_torch_to_numpy(audio_data)
    all_audio.append(audio)

# Merge and save outside the loop
merged_audio = np.concatenate(all_audio)
final_audio_path = os.path.join(output_dir, "final_story.wav")
sf.write(final_audio_path, merged_audio, sample_rate, format='WAV')
print(f" Finished! Audio saved to {final_audio_path}")


