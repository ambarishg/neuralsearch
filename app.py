import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import openai
import pandas as pd


key = "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"
collection_name = "ENMAX"


openai.api_key = key

qdrant_client = QdrantClient(host='localhost', port=6333)

def create_prompt(context,query):
    header = "Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text and requires some latest information to be updated, print 'Sorry Not Sufficient context to answer query' \n"
    return header + context + "\n\n" + query + "\n"

def generate_answer(prompt):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop = [' END']
    )
    return (response.choices[0].text).strip()

st.title("ENMAX ESG Question and Answering System")

user_input = st.text_area("Your Question",
"How do we prevent disturbance to nested birds?")
result = st.button("Make recommendations")

@st.cache_resource
def get_model():    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

if result:   
    q_new =user_input
    encoded_content = get_model().encode(q_new)
    search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=encoded_content,
            query_filter=None,  # We don't want any filters for now
            top=1  # 5 the most closest results is enough
        )
    
    payloads = [hit.payload for hit in search_result]
    metadata = [res["text"] for res in payloads]
    df = pd.DataFrame({'content': metadata })
    context= "\n\n".join(df["content"])

    context = context[:4000]
    
    prompt = create_prompt(context,q_new)
    reply = generate_answer(prompt)
    st.write(reply)