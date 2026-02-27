
import nltk
from transformers import pipeline


user = 'I how are you, I hope you are doing good. You know the movie which we saw yesterday it was good.'

# sentence_tokenizing = nltk.sent_tokenize(user)
# print(sentence_tokenizing)
# # print(emotion_model(sentence_tokenizing))
# # emotion ='joy'




class EmotionDetector():
    '''
    This class deals with converting text to different sentence tokens .Then the sentence is passed to bert model
    which will returns the emotions.
    '''
    def __init__(self,text:str):
        self.text = text 
        self.emotion_map ={
            'joy':'happy',
            'sadness':'sad',
            'anger': 'angry',
            'fear': 'fearful',
            'suprise':'exicted',
            'neutral':'calm'
        }


    def bert_model_calling(self,text:str) -> str:
        '''
        This is the function which deals with calling the bert model and returns the emotion of the sentence.
        '''
        emotion_model = pipeline(
            'text-classification',
            model='j-hartmann/emotion-english-distilroberta-base'
        )

        results_emotion_models = emotion_model(text)
        return results_emotion_models[0]['label']
    def main(self):
        '''
        This function deals with giving the emotions as a list of emotions
        '''
        sentences = nltk.sent_tokenize(self.text)
        emotion_list =[]
        for sentence in sentences:
            emotion_list.append(self.bert_model_calling(sentence))
        return emotion_list
    

# tts_emotion = emotion_map[emotion]
# sentence ='I finally got the job'

# tagged_text=f'[{tts_emotion}] {sentence}'

# aa = Emotion_detector(user)
# l = aa.main()
# print(l)