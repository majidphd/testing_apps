import streamlit as st

prompt = """
    Classify the following review
    as having either a positive or
    negative sentiment:
    
    The banana pudding was really
    tasty!
"""
response = llm_response(prompt)
print(response)

