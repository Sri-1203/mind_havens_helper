import gradio as gr
from multilingual import mono,transcribe_adio,translate_text,text_to_speech
import wave
from chat_model import model_ansing


def process_inputs(lang="en",audio_file=None):
  if lang=="Hindi":
    lang="hi-IN"
  elif lang=="Tamil":
    lang="ta-IN"
  elif lang=="Telugu":
    lang="te-IN"
  else:
    lang="en-US"

  mono(audio_file,"mono.wav")
  with wave.open("mono.wav","rb") as wav_file:
    rate=wav_file.getframerate()
    print("sample_rate:",rate)
    sample_rate=rate
  #mfcc=predict_emotion(audio_file)
  #print("mfcc",mfcc)
  text=transcribe_adio("mono.wav",sample_rate,lang)
  print("\nSpeech to text:",text)
  t_text=translate_text(text)
  print("\ntranslate text:",t_text)
  response=model_ansing(t_text)
  text=translate_text(response,lang)
  print("\nTranslate to native language:",text)
  output_audio_path=text_to_speech(text,lang)
  return t_text,response,text.strip(), output_audio_path


def process_text(lang="en-US",text=None):
  if lang=="Hindi":
    lang="hi-IN"
  elif lang=="Tamil":
    lang="ta-IN"
  elif lang=="Telugu":
    lang="te-IN"
  else:
    lang="en-US"

  t_text=translate_text(text)
  print("\ntranslate text: ",t_text)
  print(t_text)
  response=model_ansing(t_text)
  text_translate=translate_text(response,lang)
  print("\nTranslate to native language:",text_translate)
  return t_text,response,text_translate.strip(),None



def process(lang="end-US",audio_file=None,input_text=None):
  if input_text==None or input_text=="":
    return process_inputs(lang,audio_file)
  else:
    return process_text(lang,input_text)

iface = gr.Interface(
    fn=process,
    inputs=[
        gr.Dropdown(
            choices=["English", "Hindi", "Tamil", "Telugu"],  # Add more languages as needed
            label="Select Language",
            value="English"  # Default value
        ),
        gr.Audio(sources=["microphone"], type="filepath",label="Input Audio"),
        gr.Textbox(label="Text"),

    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response(ENG only)"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Response Audio", autoplay=True)

    ],
    title="psychiatrist's helper"
)

iface.launch(share=True)