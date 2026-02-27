from qwen_tts import Qwen3TTSModel
import nltk




class VoiceGenerator():
    '''
    This class deal with generating voices from the emotions along with the output of Bert model
    and concatenate different sentence audio together.
    '''
    def __init__(self,text:list,model_path:str,emotions:list):
        self.text = text
        self.emotions = emotions
        self.model = Qwen3TTSModel.from_pretrained(
            pretrained_model_name_or_path = model_path,
            local_files_only=True
        )


    def main(self):
        sentences = nltk.sent_tokenize(self.text)
        audio_arr = []
        
        for i in range(len(sentences)):
            current_emotion = self.emotions[i]
            
            # # Determine the instruction based on the previous emotion
            # if i == 0:
            #     instruction = f"Speak in a {current_emotion} tone."
            # else:
            #     previous_emotion = self.emotions[i-1]
            #     if current_emotion == previous_emotion:
            #         instruction = f"Continue speaking in a {current_emotion} tone."
            #     else:
            #         instruction = f"Make a smooth shift from {previous_emotion} to {current_emotion}."
                    
            audio, sr = self.model.generate_custom_voice(
                text=sentences[i],
                instruct=current_emotion, # pass current_emotion or instruction to get different variations
                speaker=['Ryan']
            )
            audio_arr.append((audio, sr))

        return audio_arr
