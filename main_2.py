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
    print(rate)
    sample_rate=rate
  #mfcc=predict_emotion(audio_file)
  #print("mfcc",mfcc)
  print(sample_rate)
  text=transcribe_adio("mono.wav",sample_rate,lang)
  print("got rel text")
  t_text=translate_text(text)
  print("got translate text")
  print(t_text)
  response=model_ansing(t_text)
  text=translate_text(response,lang)
  print(text)
  output_audio_path=text_to_speech(text,lang)
  print("congrats")

  return text.strip(), output_audio_path


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
  print("got translate text")
  print(t_text)
  response=model_ansing(t_text)
  text_translate=translate_text(response,lang)
  return text_translate.strip()




def process(lang="en-US", audio_file=None, input_text=None, chat_history=[]):
    if input_text is None or input_text.strip() == "":
        bot_response,bot_audio = process_inputs(lang, audio_file)  # Process speech
    else:
        bot_response = process_text(lang, input_text)  # Process text
        bot_audio=None
    
    chat_history.append((input_text if input_text else "🎤 Voice Message", bot_response))
    return chat_history, bot_audio  # Return updated chat history and response audio

with gr.Blocks() as iface:
    gr.Markdown("# 🩺 Psychiatrist's Helper - Chat Mode")

    with gr.Row():
        language_dropdown = gr.Dropdown(
            choices=["English", "Hindi", "Tamil", "Telugu"],
            label="Select Language",
            value="English"
        )
    
    chatbot = gr.Chatbot(label="Chat History")
    chat_history = gr.State([])  # Stores conversation history
    
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Speak")
        text_input = gr.Textbox(label="💬 Type Here", placeholder="Type your message...")
    send_button = gr.Button("Send",variant="primary")

    text_input.submit(
        process, 
        inputs=[language_dropdown, audio_input, text_input, chat_history], 
        outputs=[chatbot, gr.Audio(label="🔊 Response Audio", autoplay=True)])
    send_button.click(
        process, 
        inputs=[language_dropdown, audio_input, text_input, chat_history], 
        outputs=[chatbot, gr.Audio(label="🔊 Response Audio", autoplay=True)]
    )

iface.launch()