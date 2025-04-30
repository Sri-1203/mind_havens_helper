from pydub import AudioSegment
from google.cloud import translate_v2 as translate
from google.cloud import texttospeech
from google.cloud import speech_v1 as speech
import wave



client = speech.SpeechClient.from_service_account_file("./gcp_tts-cred.json")  # get form gcp
client1=texttospeech.TextToSpeechClient.from_service_account_file("./gcp_tts-cred.json") # get from gcps
def mono(in_file,out_file):
  adio=AudioSegment.from_file(in_file)
  adio=adio.set_channels(1)
  adio.export(out_file,format="wav")
  print("done converting to mono file")
  return out_file

def transcribe_adio(ifile,sample_rate,lang):
  with open(ifile, "rb") as audio_file:
    content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config=speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code=f'{lang}',
    )
    response = client.recognize(config=config, audio=audio)
    for result in response.results:
      print(result.alternatives[0].transcript,"\n")
      print(result.alternatives[0].confidence,"\n")
    return response.results[0].alternatives[0].transcript
  
def translate_text(text, target_language="en"):
    client = translate.Client()
    result = client.translate(text, target_language=target_language)
    return result["translatedText"]

def text_to_speech(text,lang):
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=f"{lang}",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client1.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    print("audio ready\n")
    output_audio_path = "response_audio.mp3"
    with open(output_audio_path, "wb") as out:
        out.write(response.audio_content)
    return output_audio_path

