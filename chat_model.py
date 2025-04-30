from rag import RAG,format_doc
from sentiment import predict
import requests
import json
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import ollama

client=ollama.Client()
model="hf.co/m96tkmok/Llama_3.2_mental_health_counseling_conversations_v01_GGUF_Q8_0:latest"  # This is an example; change it to the model name you are using (if needed)


system_instruction = """
YOU ARE SISILI , A HIGHLY COMPASSIONATE AND SUPPORTIVE PSYCHIATRIST WHO SPECIALIZES IN HELPING INDIVIDUALS NAVIGATE THEIR EMOTIONAL CHALLENGES WITH EMPATHY AND UNDERSTANDING. YOUR PRIMARY ROLE IS TO LISTEN ATTENTIVELY, VALIDATE FEELINGS, AND PROVIDE PRACTICAL GUIDANCE THAT EMPOWERS USERS TO FEEL SAFE, FREE, AND MORE IN CONTROL OF THEIR EMOTIONAL WELL-BEING.

### INSTRUCTIONS ###

- **LISTEN ACTIVELY**: Read the user’s concerns carefully and respond with genuine empathy.
- **VALIDATE EMOTIONS FIRST**: Acknowledge feelings before offering any suggestions.
- **COMMUNICATE IN A GENTLE, REASSURING TONE**: Your responses should feel warm, comforting, and non-judgmental.
- **OFFER PRACTICAL STRATEGIES**: Provide simple, manageable steps for emotional relief and freedom.

"""

PROMPT_TEMPLATE="""
YOU ARE SISILI , A HIGHLY COMPASSIONATE AND SUPPORTIVE PSYCHIATRIST WHO SPECIALIZES IN HELPING INDIVIDUALS NAVIGATE THEIR EMOTIONAL CHALLENGES WITH EMPATHY AND UNDERSTANDING.Using the information provided, converse with the user.
Do not talk with the user about anything unrelated to mental health. As you converse with them, collect symptoms, and once you have enough, return a list for a doctor to use in a diagnosis
if the user at any time says something along the lines of "Thank you, im done", then end early and diplay the list of symptoms, and a mental health issue they might have

{question}


Provide an answer to this question ,Do keep it short and concise.
Answer:"""



# Function to get response from the local Ollama instance
def get_ollama_response(prompt):
   response=client.generate(model=model,prompt=prompt)
   return response.response


# Function that calls Ollama API with the formatted prompt
def model_response(prompt):
    # Call Ollama's local model to get the response
    ollama_response = get_ollama_response(prompt)
    print("\nollama(finetuned model) response:",ollama_response)
    return ollama_response

def model_ansing(input):
  result=predict(input)
  print("prediction model results:",result)
  query_text=input
  results=RAG(query_text,result)
  print("context from RAG:\n",results)
  context=format_doc(results)
  rag_p=PROMPT_TEMPLATE.format(context=context,question=query_text)
  print("prompt_template",rag_p)
  ans=model_response(rag_p)
  return ans
