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
        esrb = game['esrb']
        rating = game['rating']
        difficulty = game['difficulty']
        length = game['length']
        description = game['description']

        # specific attributes being embedded
        embeddable_string = f"{title}{genre}{console}{developer}{esrb}{rating}{difficulty}{length}{description}"
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


def get_results(query, dev_filter, esrb_filter, console_filter=[], n_results=2, rating_minimum=0, rating_maximum=10, difficulty_minimum=0, difficulty_maximum=10, length_minimum=0, length_maximum=50):
    metadatas = []

    # adds condition of rating minimum and maximum
    where_conditions = [
        {
            "rating": {"$gte": rating_minimum}
        },
        {
            "rating": {"$lte": rating_maximum}
        },
        {
            "difficulty": {"$gte": difficulty_minimum}
        },
        {
            "difficulty": {"$lte": difficulty_maximum}
        },
        {
            "length": {"$gte": length_minimum}
        },
        {
            "length": {"$lte": length_maximum}
        }
    ]

    if dev_filter:
        where_conditions.append({"developer": {"$in": dev_filter}})

    if esrb_filter:
        where_conditions.append({"esrb": {"$in": esrb_filter}})

    if console_filter:
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


def search(query, dev_filter, esrb_filter, console_filter, n_results, rating_minimum, rating_maximum, difficulty_minimum, difficulty_maximum, length_minimum, length_maximum):
    results = get_results(query, dev_filter, esrb_filter, console_filter, n_results, rating_minimum, rating_maximum, difficulty_minimum, difficulty_maximum, length_minimum, length_maximum)

    try:
        df = pd.DataFrame(results, columns=['title', 'developer', 'genre', 'console', 'esrb', 'rating', 'difficulty', 'length', 'description'])
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
results = get_results("adventure", [], [], ["NES"], n_results=3, rating_minimum=0, rating_maximum=10, difficulty_minimum=2, difficulty_maximum=8, length_minimum=6, length_maximum=20)



for result in results:
    print(result)

# Gradio UI and deployment
with gr.Blocks() as demo:
    with gr.Tab("Game Finder"):
        with gr.Row():
            with gr.Column():
                query = gr.Textbox(label="What are you looking for in your game?", info="(Ex: sci-fi shooter, scary monsters, magic powers)", lines=5)
                dev_filter = gr.Dropdown(
                    ["Nintendo", "SquareSoft", "ChunSoft", "Capcom", "Konami"],
                    multiselect=True,
                    label="Developers",
                    info="Specific game devs? Leave blank for no filters"
                )
                esrb_filter = gr.CheckboxGroup(
                    ["EVERYONE", "EVERYONE 10+", "TEEN", "MATURE 17+", "ADULTS ONLY 18+"],
                    label="ESRB Rating",
                    info="Age restrictions? Leave blank for no filters"
                )
                console_filter = gr.CheckboxGroup(
                    ["NES"],
                    label="Consoles",
                    info="Specific consoles only? Leave blank for no filters"
                )
                with gr.Row():
                    rating_minimum = gr.Slider(label="Minimum rating", minimum=0, maximum=10, value=0, step=1)
                    rating_maximum = gr.Slider(label="Maximum rating", minimum=0, maximum=10, value=10, step=1)
                with gr.Row():
                    difficulty_minimum = gr.Slider(label="Minimum difficulty", minimum=0, maximum=10, value=0, step=1)
                    difficulty_maximum = gr.Slider(label="Maximum difficulty", minimum=0, maximum=10, value=10, step=1)
                with gr.Row():
                    length_minimum = gr.Slider(label="Minimum hours", minimum=0, maximum=50, value=0, step=1, info="The shortest game in the database is -")
                    length_maximum = gr.Slider(label="Maximum hours", minimum=0, maximum=50, value=50, step=1, info="The longest game in the database is -")
                n_results = gr.Slider(label="Results to Display", info="If there are not enough results, it will display what is available", minimum=0, maximum=10, value=2, step=1)
                btn = gr.Button(value="Submit")
                table = gr.Dataframe(label="Results", headers=['title', 'developer', 'genre', 'console', 'esrb', 'rating', 'difficulty', 'length', 'description'], wrap=True)
            btn.click(search, inputs=[query, dev_filter, esrb_filter, console_filter, n_results, rating_minimum, rating_maximum, difficulty_minimum, difficulty_maximum, length_minimum, length_maximum], outputs=[table])
    demo.launch(share=True)
