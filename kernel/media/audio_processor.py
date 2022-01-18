import pyttsx3
engine = pyttsx3.init()


def speak(text):
    """
    Speak text.

    @param text: input text message
    """
    while True:
        if len(text):
            engine.say(text[-1])
            text.pop()
            engine.runAndWait()
