import speech_recognition as sr

from langgraph.checkpoint.mongodb import MongoDBSaver 

MONGODB_URI = "mongodb://localhost:27017/langgrah"
config = {"configurable": {"thread_id": "5"}}
from graph import create_chat_graph


def main():
    
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph = create_chat_graph(checkpointer=checkpointer)
        r = sr.Recognizer() 
        r.pause_threshold = 1 #will stop detecitng audio after last sound is made
        with sr.Microphone() as source:
            # r.adjust_for_ambient_noise()
            print("Say something:")
            audio = r.listen(source)
            print("Processing Audio")
            sst = r.recognize_google(audio, language='en-US')
            print("Text Said: ", sst)
            for event in graph.stream({"messages" : [{"role": "user", "content": sst}]}, config, stream_mode="values"):
                if "messages" in event:
                    event["messages"][-1].pretty_print()
main()



#now we can langchain to feed our text to gpt