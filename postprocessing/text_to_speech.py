
from google.cloud import speech_v1p1beta1 as speech
import anthropic
import json
import requests
# == VIDEO TO TEXT -- NEED GOOGLE CLOUD API
with open("secrets.json", "r", encoding="utf-8") as file:
    secrets = json.load(file)
    GOOGLE_CLOUD_API_KEY = secrets["google_cloud_api_key"]
    ANTHROPIC_API_KEY = secrets["anthropic_api_key"]
    ELEVEN_LABS_API_KEY = secrets["eleven_labs_api_key"]
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


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


def text_to_speech(json_data):
    prompt = f"""Summarize the following json data using plain sentences
    data : {json_data}. JUST WRITE A SUMMARY, NOTHING ELSE"""
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    summary = response.content[0].text
    # The text you want to convert to speech

    # Construct the API endpoint URL
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

    # Define the request headers and payload
    headers = {
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json"
    }

    data = {
        "text": summary,
        "voice_settings": {
            "stability": 0.5,         # Adjust stability as needed (0 to 1)
            "similarity_boost": 0.75    # Adjust similarity boost as needed (0 to 1)
        }
    }

    # Make the POST request to the ElevenLabs API
    response = requests.post(url, json=data, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the binary response content as an MP3 file
        output_filename = "output.mp3"
        with open(output_filename, "wb") as f:
            f.write(response.content)
        print(f"MP3 file saved as {output_filename}")
    else:
        print("Error:", response.status_code, response.text)
    
    return output_filename

