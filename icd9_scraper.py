import requests
import os
import pandas as pd
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

def scrape(url, filename, label=''):
    all_icd9 = []
    r = requests.get(url)
    assert r.ok
    df = pd.read_html(r.content)[0]
    converter = {c:str for c in df.columns}

    def scrape_util(sub_url='', contents=[]):
        r = requests.get(urljoin(url, sub_url))
        assert r.ok
        try:
            df = pd.read_html(r.content, converters=converter)[0]
            for _, row in df.iterrows():
                group = row['ICD Code']; description = row['Description']
                contents.append(description)
                scrape_util(group, contents)
                contents.pop()
        except (KeyError, ImportError):
            pass
        if sub_url and contents and '-' not in sub_url:
            print(label, len(all_icd9), sub_url, contents[-1]) # for debugging
            new_data = {'code': sub_url}
            for i in range(len(contents)):
                new_data[f'description_{i+1}'] = contents[i]
            all_icd9.append(new_data)

    scrape_util()
    all_icd9_df = pd.DataFrame(all_icd9)
    all_icd9_df.sort_values(by='code').to_csv(filename, index=False)
    return all_icd9_df

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [
        executor.submit(scrape, 'https://dexur.com/icd9/', os.path.join(os.getcwd(), 'processed_data', 'icd9_diagnosis_backup.csv'), 'diagnosis'),
        executor.submit(scrape, 'https://dexur.com/pcs9/', os.path.join(os.getcwd(), 'processed_data', 'icd9_procedure_backup.csv'), 'procedure')
    ]
    for f in as_completed(futures):
        print(f.result())
