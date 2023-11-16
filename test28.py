import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("huggingface/conversational-ai-checkpoint")
model = AutoModelForQuestionAnswering.from_pretrained("huggingface/conversational-ai-checkpoint")

# Define the Streamlit app
st.title("LLM-powered Q&A")

# Get the user's question
question = st.text_input("Ask a question:")

# Convert the question to tokens
input_ids = tokenizer(question, return_tensors="pt")

# Generate the answer
outputs = model(**input_ids)
pred_start_position, pred_end_position = outputs.start_logits.argmax(-1).item(), outputs.end_logits.argmax(-1).item()
answer = tokenizer.decode(outputs.input_ids[0][pred_start_position:pred_end_position+1])

# Display the answer
st.write(answer)
