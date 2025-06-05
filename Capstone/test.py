import requests
import os
import pandas as pd
import time
from tqdm import tqdm
from pprint import pprint

def get_tumor_code(case_id):
    """
    Fetches the tumor code for a given case ID from the GDC API.

    Args:
        case_id (str): The case ID to query.

    Returns:
        str or None: The tumor code if found, otherwise None.
    """
    url = f"https://api.gdc.cancer.gov/files/{case_id}?expand=cases"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['data']['cases'][0]['disease_type'] == "Myeloid Leukemias":
            return "Acute myeloid leukemia (AML)"
        elif data['data']['cases'][0]['disease_type'] == "Lymphoid Leukemias":
            return "Acute lymphoblastic leukemia (ALL)"
        else:
            print(f"Case ID {case_id} does not match the expected disease type.")
            print(data['data']['cases'][0]['disease_type'])
        return None
    except Exception as e:
        print("You shit the bed, something went wrong with the API request.")
        print(e)
        return None

def main():
    checked_ids_file = "disease_ids.csv"
    if not os.path.exists(checked_ids_file):
        with open(checked_ids_file, 'w') as f:
            f.write("case_id,tumor_code\n")

    df = pd.read_csv(checked_ids_file)
    df = df.drop_duplicates(subset='case_id', keep='last')
    df.to_csv(checked_ids_file, index=False)
    case_ids_to_check = []

    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith(".maf.gz"):
                case_id = root.split(os.sep)[-1]
                if case_id not in df['case_id'].values or pd.isnull(df.loc[df['case_id'] == case_id, 'tumor_code']).any():
                    case_ids_to_check.append(case_id)

    for case_id in case_ids_to_check:
        tumor_code = get_tumor_code(case_id)
        time.sleep(0.2)
        print(f"Case ID: {case_id}, Tumor Code: {tumor_code}")
        df2 = pd.DataFrame({'case_id': [case_id], 'tumor_code': [tumor_code]})
        df = pd.concat([df, df2], ignore_index=True)

    df = df.drop_duplicates(subset='case_id', keep='last')
    df.to_csv(checked_ids_file, index=False)

    print(f"Updated {checked_ids_file} with new case IDs and tumor codes.")

if __name__ == "__main__":
    main()
