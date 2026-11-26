import alpha_mining.lib as ace
import helpful_functions as hf
import pandas as pd
import plotly.express as px
import time
from tqdm import tqdm
import streamlit as st
from pathlib import Path
import os
from .core import region_config

if 'alpha' not in st.session_state:
    class AlphaState:
        def __init__(self):
            self.s = None
            self.ds = pd.DataFrame()
            self.df = pd.DataFrame()
            self.id = []
            self.region = None
            self.universe = None
            self.delay = None
            self.alphas = {}
            self.op = []
    st.session_state.alpha = AlphaState()

# Login and reset session
def login(): 
    st.session_state.alpha.s = ace.start_session() 
    ace.check_session_and_relogin(st.session_state.alpha.s)
    print("Login successful!")

def has_saved_credentials():
    credentials_file_path = os.path.join(os.path.expanduser("~"), "secrets", "platform-brain.json")
    return Path(credentials_file_path).exists() and os.path.getsize(credentials_file_path) > 2

# Selecting datasets
def fetch_dataset(
        delay,
        region, 
        universe, 
        category_id=None, 
        coverage=None, 
        field_count=None, 
        user_count=None, 
        value_score=None):
    if st.session_state.alpha.s is None:
        st.error("Please log in first!")
        return
    
    st.session_state.alpha.region = region
    st.session_state.alpha.universe = universe

    datasets = ace.get_datasets(    
        st.session_state.alpha.s,
        instrument_type='EQUITY',
        region=region,
        delay=delay,
        universe=universe,
        theme='false'
    )

    if not isinstance(datasets, pd.DataFrame):
        datasets = pd.DataFrame(datasets)

    mask = pd.Series(True, index=datasets.index)

    if coverage and len(coverage) == 2 and 'coverage' in datasets.columns:
        mask &= datasets['coverage'].between(coverage[0], coverage[1])

    if field_count and len(field_count) == 2 and 'fieldCount' in datasets.columns:
        mask &= datasets['fieldCount'].between(field_count[0], field_count[1])

    if user_count and len(user_count) == 2 and 'userCount' in datasets.columns:
        mask &= datasets['userCount'].between(user_count[0], user_count[1])

    if value_score and len(value_score) == 2 and 'valueScore' in datasets.columns:
        mask &= datasets['valueScore'].between(value_score[0], value_score[1])

    if category_id:
        if 'category_id' in datasets.columns:
            mask &= datasets['category_id'] == category_id
        elif 'category' in datasets.columns:
            mask &= datasets['category'] == category_id

    selected_datasets = datasets[mask].sort_values(by=['valueScore'] if 'valueScore' in datasets.columns else datasets.columns[0], ascending=False)
    selected_datasets = selected_datasets.reset_index(drop=True)

    st.session_state.alpha.ds = selected_datasets

    if 'id' in selected_datasets.columns:
        st.session_state.alpha.id = selected_datasets['id'].tolist()
    else:
        st.session_state.alpha.id = []

# Fetch datafields
def fetch_datafields():
    dfs = []
    for dataset_id in st.session_state.alpha.id:
        df = ace.get_datafields(
            st.session_state.alpha.s,
            instrument_type='EQUITY',
            region = st.session_state.alpha.region,
            delay=st.session_state.alpha.delay,
            universe=st.session_state.alpha.universe,
            theme='false', 
            search=dataset_id   
        )
        if not df.empty:
            dfs.append(df)
        time.sleep(0.5)
    
    # Safely concat all fetched datasets
    if dfs:
        st.session_state.alpha.df = pd.concat(dfs, ignore_index=True)
    else:
        st.session_state.alpha.df = pd.DataFrame()
        
    st.dataframe(st.session_state.alpha.df) 

def create_expressions(category, look_back_time):
    ops = ace.get_operators(st.session_state.alpha.s)
    
    # Expressions logic
    matrix_ops = ops[(ops['category'] == 'MATRIX') & (ops['scope'] == 'REGULAR')]
    vector_ops = ops[(ops['type'] == 'VECTOR') & (ops['scope'] == 'REGULAR')]
    
    all_expressions = []
    
    if category == 'MATRIX':
        matrix_df = st.session_state.alpha.df[st.session_state.alpha.df['type'] == 'MATRIX']
        # Iterate over the ID column, not the DataFrame itself
        for op in matrix_ops['id']: 
            expressions = [f'{op}({x},{look_back_time})' for x in matrix_df.id.values.tolist()]
            all_expressions.extend(expressions)
    else:
        vector_df = st.session_state.alpha.df[st.session_state.alpha.df['type'] == 'VECTOR']
        for v_op in vector_ops['id']:
            for m_op in matrix_ops['id']:
                expressions = [f'{m_op}({v_op}({x}),{look_back_time})' for x in vector_df.id.values.tolist()]
                all_expressions.extend(expressions)
                
    expression_list = pd.DataFrame(all_expressions, columns=['expression'])
    return expression_list

if __name__ == '__main__':
    
    st.title('World Quant Brain automatic alpha miner of vdhwy')

    if st.session_state.alpha.s is None:
        st.warning("Please log in to continue.")
        st.button(
            label='Log into your account',
            disabled=False,
            type='primary',
            on_click=login
        )
        st.stop()
    else:
        st.success("Logged in securely")

    st.session_state.alpha.delay = st.selectbox("Delay", [0,1]) 
    valid_regions = set()
    for x in UNIVERSES:
        delays = x['delay'] if isinstance(x['delay'], list) else [x['delay']]
        if st.session_state.alpha.delay in delays:
            valid_regions.add(x['region'])
    filtered_regions = [r for r in REGIONS if r in valid_regions]
    st.session_state.alpha.region = st.selectbox("Region", filtered_regions, index=None, placeholder="Select Region...")
    
    universes = []
    for i in UNIVERSES:
        if st.session_state.alpha.region is None or i['region'] == st.session_state.alpha.region:
            if isinstance(i['name'], list):
                universes.extend(i['name'])
            else:
                universes.append(i['name'])
    universes = sorted(list(set(universes)))
            
    st.session_state.alpha.universe = st.selectbox("Universe", universes if universes else ['N/A'])
    
    coverage = st.slider("Coverage", min_value=0.0, max_value=1.0, value=(0.0, 1.0), format="%.2f") 
    field_count = st.slider("Field Count", min_value=0, max_value=1000, value=(1, 999), step=1)
    user_count = st.slider("User Count", min_value=0, max_value=1000, value=(1, 999), step=1)
    value_score = st.slider("Value Score", min_value=0.0, max_value=10.0, value=(0.1, 9.9), format="%.1f")

    st.button(
        label='Find datasets',
        disabled=False,
        on_click=fetch_dataset,
        kwargs={
            'delay': st.session_state.alpha.delay,
            'region': st.session_state.alpha.region,
            'universe': st.session_state.alpha.universe,
            'category_id': None, 
            'coverage': coverage,
            'field_count': field_count,
            'user_count': user_count,
            'value_score': value_score
        }
    )
    
    # Dynamic Rendering Logic
    if not st.session_state.alpha.ds.empty:
        # Dynamic Text Filter for Dataset ID
        filter_id = st.text_input('Filter by Dataset ID:', placeholder="Type to instantly filter...")
        display_ds = st.session_state.alpha.ds.copy()
        # Apply filter if user typed something
        if filter_id:
            display_ds = display_ds[display_ds['id'].astype(str).str.contains(filter_id.strip(), case=False, na=False)]
            st.session_state.alpha.id = display_ds['id'].tolist()
            
        st.write(f"### Filtered Results ({len(display_ds)} rows)")
        st.dataframe(display_ds)

        st.markdown('Filter your datafields')
        st.button(
            label='Filter datafields',
            disabled=False,
            type='primary',
            on_click=fetch_datafields
        )
    if not st.session_state.alpha.df.empty:
        category = st.selectbox('Select dataframe type', ['MATRIX', 'VECTOR'])
        
        if category:
            # Render the input in the main layout, NOT in the callback
            look_back = st.text_input('Input look back time')
            
            # Use a standard button check instead of on_click
            if st.button(label='Create expression', type='primary'):
                if look_back:
                    result_df = create_expressions(category, look_back)
                    st.success(f"Generated {len(result_df)} expressions!")
                    st.dataframe(result_df)
                else:
                    st.warning("Please input a look back time first.")
                

    