import faiss
import ollama
from tqdm import tqdm
import numpy as np
import pandas as pd

# ========== Embedding Function ==========
def encode(text):
    return ollama.embeddings(model='nomic-embed-text', prompt=text)['embedding']

# ========== Load and Index Traffic Language Module (TLM) ==========
df_lang = pd.read_excel('TLM.xlsx', header=0)
traffic_lang_data = []
for _, row in df_lang.iterrows():
    scenario = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
    language = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ""
    traffic_lang_data.append({
        'scenario': scenario.strip(),
        'standard_language': language.strip()
    })

lang_embeddings = [encode(item['scenario']) for item in tqdm(traffic_lang_data, desc='Embedding TLM')]
lang_embeddings = np.array(lang_embeddings).astype('float32')
faiss.normalize_L2(lang_embeddings)
lang_index = faiss.index_factory(lang_embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
lang_index.add(lang_embeddings)

# ========== Load CAM by Control Algorithm ==========
def load_cam_by_algorithm(algo_name: str):
    filename = f"{algo_name.upper()}_CAM.xlsx"
    try:
        df_knowledge = pd.read_excel(filename, header=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"Strategy file '{filename}' not found.")
    
    strategy_knowledge = []
    for _, row in df_knowledge.iterrows():
        language = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        input_condition = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        format_desc = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
        strategy = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ""
        strategy_knowledge.append({
            'language': language,
            'input': input_condition,
            'format': format_desc,
            'strategy': strategy
        })

    strategy_embeddings = [encode(item['language']) for item in tqdm(strategy_knowledge, desc=f'Embedding CAM - {algo_name.upper()}')]
    strategy_embeddings = np.array(strategy_embeddings).astype('float32')
    faiss.normalize_L2(strategy_embeddings)
    strategy_index = faiss.index_factory(strategy_embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    strategy_index.add(strategy_embeddings)

    return strategy_knowledge, strategy_index

# ========== Main Query Function ==========
def query_strategy_from_event(event_description: str, control_algorithm: str = "MPC") -> dict:
    """
    Given a traffic event description and control algorithm name,
    return standard traffic language, input condition, format requirement, and strategy.
    """
    # Phase 1: retrieve traffic language
    event_vec = np.array([encode(event_description)]).astype('float32')
    faiss.normalize_L2(event_vec)
    _, lang_indices = lang_index.search(event_vec, k=1)
    standard_lang = traffic_lang_data[lang_indices[0][0]]['standard_language']

    # Phase 2: load CAM and retrieve strategy
    strategy_knowledge, strategy_index = load_cam_by_algorithm(control_algorithm)
    lang_vec = np.array([encode(standard_lang)]).astype('float32')
    faiss.normalize_L2(lang_vec)
    _, strategy_indices = strategy_index.search(lang_vec, k=1)
    strategy_data = strategy_knowledge[strategy_indices[0][0]]

    return {
        'event_description': event_description,
        'standard_language': standard_lang,
        'input_condition': strategy_data['input'],
        'format_requirement': strategy_data['format'],
        'strategy': strategy_data['strategy']
    }

# ========== Interactive Function ==========
def interactive_query():
    """
    Prompt user to input traffic event and control algorithm, 
    handle errors if the CAM file is not found, and allow re-entry or exit.
    """
    while True:
        event = input("Enter traffic event description: ").strip()
        control_algo = input("Enter control algorithm name (e.g., MPC, MaxPressure): ").strip()

        if not event or not control_algo:
            print("Both traffic event and control algorithm name are required.")
            return

        try:
            result = query_strategy_from_event(event, control_algo)
            print("\n=== Retrieval Result ===")
            print("Event Description:", result['event_description'])
            print("Standard Traffic Language:", result['standard_language'])
            print("Input Condition:", result['input_condition'])
            print("Format Requirement:", result['format_requirement'])
            print("Strategy:", result['strategy'])
            return result

        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
            retry = input("Do you want to try again? (y/n or 'q' to quit): ").strip().lower()
            if retry != 'y':
                print("Exiting program.")
                break

# ========== Entry Point ==========
#if __name__ == "__main__":
#    result = interactive_query()
