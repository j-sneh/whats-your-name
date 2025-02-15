import whisper

model = whisper.load_model("base")
result = model.transcribe("Hello.mp3")
print(result["text"])
print(result)