# import gtts
# tts = gtts.gTTS("Hi Bixby",slow=False)
# tts.save('voice/'+'hi_bixby.wav')

##################################
# import pyttsx3
# # initialize Text-to-speech engine
# # engine = pyttsx3.init()
# # convert this text to speech
# text = "Hi Bixby"
# engine = pyttsx3.init("sapi5")
# rate = engine.getProperty("rate")
# print(rate)
# engine.setProperty("rate", 150)
# engine.say(text)
# # play the speech
# # engine.save_to_file(text, 'hi.wav')
# engine.runAndWait()

######################################
#
# from ibm_watson import TextToSpeechV1
# from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
#
# authenticator = IAMAuthenticator('{apikey}')
# text_to_speech = TextToSpeechV1(
#     authenticator=authenticator
# )
#
# text_to_speech.set_service_url('https://api.us-east.text-to-speech.watson.cloud.ibm.com')