import streamlit as st
from transformers import pipeline

# Load the RAG model for text generation
rag_generator = pipeline("text-generation", model="facebook/rag-token-nq")

def generate_response(prompt):
    # Generate a response using the RAG model
    response = rag_generator(prompt, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']

def main():
    st.title("RAG (Retrieval-Augmented Generator) Streamlit App")
    st.write("Enter a prompt and let RAG generate a response!")

    # User input for the prompt
    user_prompt = st.text_area("Enter Prompt:", "")

    if st.button("Generate Response"):
        if user_prompt:
            # Generate response and display it
            generated_response = generate_response(user_prompt)
            st.write("Generated Response:")
            st.write(generated_response)
        else:
            st.warning("Please enter a prompt.")

if __name__ == "__main__":
    main()
