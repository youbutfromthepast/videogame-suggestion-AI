import chromadb
import gradio as gr
import json
import openai
import os
import pandas as pd

from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------METHODS----------------------------------------------
def load_data():
    with open("data.json") as f:
        data = json.load(f)
    return data


def get_chroma_collection(collection_name):
    chroma_client = chromadb.PersistentClient(path=".")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    api_base=os.getenv("OPENAI_API_BASE"),
                    model_name="text-embedding-ada-002"
                )

    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=openai_ef)
    return collection


def add_data_to_collection(data, collection):
    documents = []
    metadatas = []
    ids = []

    for i, game in enumerate(data):
        # grabs attributes from data.json
        title = game['title']
        genre = game['genre']
        console = game['console']
        developer = game['developer']
        # esrb = game['esrb']
        rating = game['rating']
        # difficulty = game['difficulty']
        # length = game['length']
        description = game['description']

        # specific attributes being embedded
        embeddable_string = f"{title}{genre}{console}{developer}{rating}{description}"
        documents.append(embeddable_string)

        # stores as metadata
        metadatas.append(game)

        # index as id
        ids.append(str(i))

    # updates collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )


def get_results(query, dev_filter, console_filter=[], n_results=2, rating_minimum=0, rating_maximum=10):
    metadatas = []

    # adds condition of rating minimum and maximum
    where_conditions = [
        {
            "rating": {"$gte": rating_minimum}
        },
        {
            "rating": {"$lte": rating_maximum}
        }
    ]

    if dev_filter:
        where_conditions.append({"developer": {"$in": dev_filter}})

    if console_filter:
        print(console_filter)
        where_conditions.append({"console": {"$in": console_filter}})

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"$and": where_conditions}
    )

    num_results = len(results["metadatas"][0])
    for i in range(min(n_results, num_results)):
        metadatas.append(results["metadatas"][0][i])

    return metadatas


def search(query, dev_filter, console_filter, n_results, rating_minimum, rating_maximum):
    results = get_results(query, dev_filter, console_filter, n_results=n_results, rating_minimum=rating_minimum, rating_maximum=rating_maximum)

    try:
        df = pd.DataFrame(results, columns=['title', 'genre', 'console', 'developer', 'rating', 'description'])
        return df
    except Exception as e:
        raise gr.Error(e.message)


# ----------------------------------------------MAIN----------------------------------------------

# definitions
data = load_data()
collection = get_chroma_collection("games")

# add data to chroma collection
add_data_to_collection(data, collection)

# result check
results = get_results("adventure", ["Capcom", "ChunSoft", "Nintendo"], ["Super Nintendo"], n_results=3, rating_minimum=0, rating_maximum=10)



for result in results:
    print(result)

# Gradio UI and deployment
with gr.Blocks() as demo:
    with gr.Tab("Game Finder"):
        with gr.Row():
            with gr.Column():
                query = gr.Textbox(label="What are you looking for?", lines=5)
                dev_filter = gr.Dropdown(
                    ["Nintendo", "SquareSoft", "ChunSoft", "Capcom", "Konami"],
                    multiselect=True,
                    label="Preferred developers?"
                )
                console_filter = gr.CheckboxGroup(
                    ["NES", "Super Nintendo", "Playstation", "Dreamcast", "Playstation 3", "Xbox 360", "Playstation 4", "Nintendo Switch", "PC", "Game Boy", "Game Boy Color", "Game Boy Advance", "DS", "PSP", "3DS", "iOS", "Android"],
                    label="Console filter",
                    info="Specific consoles only?"
                )
                with gr.Row():
                    rating_minimum = gr.Slider(label="Minimum rating", minimum=0, maximum=10, value=0, step=1)
                    rating_maximum = gr.Slider(label="Maximum rating", minimum=0, maximum=10, value=10, step=1)
                n_results = gr.Slider(label="Results to Display", minimum=0, maximum=10, value=2, step=1)
                btn = gr.Button(value="Submit")
                table = gr.Dataframe(label="Results", headers=['title', 'genre', 'console', 'developer', 'rating', 'description'], wrap=True)
            btn.click(search, inputs=[query, dev_filter, console_filter, n_results, rating_minimum, rating_maximum], outputs=[table])
    demo.launch(share=True)
