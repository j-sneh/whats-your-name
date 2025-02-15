
from google.cloud import speech_v1p1beta1 as speech
import anthropic
# == VIDEO TO TEXT -- NEED GOOGLE CLOUD API
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def transcribe_speech(speech_uri) -> str:
    # If you're using ADC, no need to pass client_options
    client = speech.SpeechClient()

    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=2,
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        model="phone_call",
        diarization_config=diarization_config,
        api_key = GOOGLE_CLOUD_API_KEY
    )


    
    audio = speech.RecognitionAudio(uri=speech_uri)
    
    response = client.recognize(config=config, audio=audio)
    
    transcript = " ".join([result.alternatives[0].transcript for result in response.results])
    
    return transcript



def extract_info_claude(transcript):
    prompt = f"""
    You are a professional conversationalist, and you are very good at making summaries. 
    You will be given a transcription between speaker 1 and speaker 2. Then, you will be given a set of words, each being annotated
    with whether they belong to speaker 1 or speaker 2. The first to say "my friend" is the user.
    Extract the following information from the speaker that is not face blind. Also, give the timestamp in which the interlocutor 
    gives out their names. 
    Transcript:
    {transcript}
    The 'Other Infos' part should contain at most 5 short bullet points 
    Provide the response in this JSON format:
    {{
        "name": "[Insert the name here]",
        "other_info": "Insert the bullet points here"
        "timestamp : [Insert the timestamp in seconds here]"
    }}
    DO NOT WRITE ANYTHING OUTSIDE OF BRACKETS
    """

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text

