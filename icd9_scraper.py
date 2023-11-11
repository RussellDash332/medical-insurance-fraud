import requests
import os
import pandas as pd
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

def new_get(url, sub_url, contents):
    print(sub_url, contents) # for debugging
    r = requests.get(url); assert r.ok
    return (r, sub_url, contents)

def scrape(url, filename, label=''):
    all_icd9 = []
    r = requests.get(url); assert r.ok
    df = pd.read_html(r.content)[0]
    converter = {c:str for c in df.columns}

    queue = [('', [])]
    while queue:
        new_queue = []
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(new_get, urljoin(url, sub_url), sub_url, contents) for sub_url, contents in queue]
        for f in as_completed(futures):
            r, sub_url, contents = f.result()
            try:
                df = pd.read_html(r.content, converters=converter)[0]
                for _, row in df.iterrows():
                    group = row['ICD Code']; description = row['Description']
                    new_queue.append((group, contents+[description]))
            except (KeyError, ImportError):
                pass
            if sub_url and contents and '-' not in sub_url:
                new_data = {'code': sub_url}
                for i in range(len(contents)):
                    new_data[f'description_{i+1}'] = contents[i]
                all_icd9.append(new_data)
        queue = new_queue

    all_icd9_df = pd.DataFrame(all_icd9)
    all_icd9_df.sort_values(by='code').to_csv(filename, index=False)
    return all_icd9_df

scrape('https://dexur.com/icd9/', os.path.join(os.getcwd(), 'processed_data', 'icd9_diagnosis.csv'), 'diagnosis')
scrape('https://dexur.com/pcs9/', os.path.join(os.getcwd(), 'processed_data', 'icd9_procedure.csv'), 'procedure')