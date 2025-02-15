from google.cloud import speech_v1p1beta1 as speech
import sys
import json

if len(sys.argv) < 2:
    print("Please provide the audio file name as a command-line argument.")
    sys.exit(1)

speech_file = sys.argv[1]

with open("secrets.json", "r", encoding="utf-8") as file:
    secrets = json.load(file)
    GOOGLE_CLOUD_API_KEY = secrets["google_cloud_api_key"]
    ANTHROPIC_API_KEY = secrets["anthropic_api_key"]
    ELEVEN_LABS_API_KEY = secrets["eleven_labs_api_key"]


client = speech.SpeechClient(client_options={"api_key": GOOGLE_CLOUD_API_KEY })

with open(speech_file, "rb") as audio_file:
    content = audio_file.read()

audio = speech.RecognitionAudio(content=content)

diarization_config = speech.SpeakerDiarizationConfig(
    enable_speaker_diarization=True,
    min_speaker_count=2,
    max_speaker_count=10,
)

config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=8000,
    language_code="en-US",
    diarization_config=diarization_config,
)

print("Waiting for operation to complete...")
response = client.recognize(config=config, audio=audio)
print(response)


if len(response.results) < 1:
    print("No results found")
    sys.exit(1)

result = response.results[-1]
print(result)
words_info = result.alternatives[0].words

# Printing out the output:
for word_info in words_info:
    print(f"word: '{word_info.word}', speaker_tag: {word_info.speaker_tag}")
