#coding:UTF-8
from fastapi import FastAPI,HTTPException
import re
from transformers import BertForSequenceClassification,AutoTokenizer,Trainer,BartTokenizer, BartForConditionalGeneration
import torch
import requests
import speech_recognition as sprc
import json
import base64
import io
from pydantic import BaseModel

app=FastAPI()


def arabizi_transliterate(text):
    
    translit_dict = {
        'ب': 'b',
        'ت': 't',
        'ث': 'th',
        'ج': 'j',
        'ح': 'h',
        'خ': 'kh',
        'د': 'd',
        'ذ': 'dh',
        'ر': 'r',
        'ز': 'z',
        'س': 's',
        'ش': 'sh',
        'ص': 's',
        'ض': 'd',
        'ط': 't',
        'ظ': 'dh',
        'ع': "3",
        'غ': 'gh',
        'ف': 'f',
        'ق': '9',
        'ك': 'k',
        'ل': 'l',
        'م': 'm',
        'ن': 'n',
        'ه': 'h',
        'و': 'w',
        'ي': 'y',
        'ء': "a",
        'آ': 'aa',
        'أ': 'a',
        'إ': 'i',
        'ئ': 'e',
        'ة': 't',
        'ى': 'a',
        'ؤ': 'o',
        'أ': 'a',
        'لأ': 'la', 
        'لإ': 'li', 
        'لآ': 'la',  
        'لأ': 'la',
        'ا': 'a',
        'َ': '',
        'ُ': '',
        'ِ': 'i',
        'ّ': '',
        'ْ': '',
    }

    transliterated = ""
    skip_next = False

    for i in range(len(text)):
        if skip_next:
            skip_next = False
            continue

        char = text[i]

        if i + 1 < len(text):
            combo = char + text[i + 1]
            if combo in translit_dict:
                transliterated += translit_dict[combo]
                skip_next = True
                continue

        transliterated += translit_dict.get(char, char)

    return transliterated


class RecognizeData(BaseModel):
    a:str
    
def base64_to_wav(str64):
    audio_bytes=base64.b64decode(str64)
    return io.BytesIO(audio_bytes)

def recognize_audio_ar(wave_data):
    recognizer=sprc.Recognizer()
    audio_data=sprc.AudioData(wave_data.read(),sample_rate=44100,sample_width=2)
    text=""
    try:
        text=recognizer.recognize_google(audio_data,language="ar-DZ")
        text = arabizi_transliterate(text)
    except:
        pass
    return text

class CoherenceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]) if self.labels else None
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
    
DarijaBert_model = BertForSequenceClassification.from_pretrained('./CohereZiClass')
DarijaBERT_tokenizer = AutoTokenizer.from_pretrained('./CohereZiClass')
trainer = Trainer(model=DarijaBert_model)
device = torch.device("cpu")
DarijaBert_model.to(device)

tokenizer = BartTokenizer.from_pretrained('./CohereZiGenerator')
model = BartForConditionalGeneration.from_pretrained('./CohereZiGenerator')
model.to(device)

def text_a_phrases(text):
    return [s.strip() for s in re.split(r"[.,;!?]", text) if s.strip()]
    

def transformer_caracteres_repetes(texte):
    return re.sub(r"(a)\1{3,}",r"\1\1",re.sub(r"([^a])\1+", r"\1", texte))

def enlever_les_caractaire_special(text):
    text = re.sub(r"é|è|ê|ë",r"e", text)
    text = re.sub(r"á|à|â|ä",r"a", text)
    text = re.sub(r"ó|ò|ô|ö",r"o", text)
    text = re.sub(r"ú|ù|û|ü",r"u", text)
    text = re.sub(r"í|ì|î|ï",r"i", text)
    return re.sub(r"ý|ỳ|ŷ|ÿ",r"y", text)


@app.post("/audio")
async def recognize_audio_endpoint(data:RecognizeData):
    wave_data=base64_to_wav(data.a)
    recognition_result=recognize_audio_ar(wave_data)
    print(recognition_result)
    return recognition_result
    try:
        pass
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Error processing audio: {str(e)}")

@app.get("/coherance")
async def process_rasa_message(message:str):
    try:
        message = message.lower()
        message = transformer_caracteres_repetes(message)
        message = enlever_les_caractaire_special(message)
        phrase = text_a_phrases(message)

        test_encodings = DarijaBERT_tokenizer(phrase, truncation=True, padding=True, max_length=256)
        
        test_dataset = CoherenceDataset(test_encodings, labels=None)
        
        
        predictions = trainer.predict(test_dataset)
        
        probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions.predictions), dim=-1)
        
        predicted_labels = torch.argmax(probabilities, dim=1)
        coherent = predicted_labels.tolist()

        suggestion = []

        for i in range(len(phrase)):
            if coherent[i] == 0:
                sug = []
                for _ in range(2):
                    input_ids = tokenizer.encode(phrase[i], return_tensors="pt")
                    output_ids = model.generate(input_ids, max_length=20, do_sample=True)
                    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    sug.append(output_text)
                suggestion.append(sug)
            else:
                suggestion.append([])
        

        phrase_arab = []
        for a in phrase:
            phrase_arab.append(requests.get(f"http://127.0.0.1:3000/message?msg={a}").text)
        
        suggestion_arab = []
        for b in suggestion:
            sug = []
            for c in b:
                sug.append(requests.get(f"http://127.0.0.1:3000/message?msg={c}").text)
            suggestion_arab.append(sug)
        
        phrase = phrase + phrase_arab
        coherent = coherent + coherent
        suggestion = suggestion + suggestion_arab
        
        phrases = [ { "text" : phrase[i] , "est_coherent": coherent[i] == 1 , "suggestion": suggestion [i] } for i in range(len(phrase)) ]
        
        return {"role":"assistant", "phrases": phrases }
    except Exception as e:
        raise HTTPException(status_code=500,detail="Erreur lors du traitement de la requete : "+str(e))

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=5000)