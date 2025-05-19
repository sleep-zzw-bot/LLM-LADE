import os
import random

import pandas as pd
import json

from sklearn.model_selection import train_test_split

def extract_columns(input_file, output_file):
    # Read the file with pandas, handling null bytes gracefully
    df = pd.read_csv(input_file, encoding='utf-8', error_bad_lines=False)
    df['Label'] = df['Label'].apply(lambda x: 'normal' if x == '-' else 'abnormal')
    # Extract the needed columns
    new_df = df[['Label', 'EventTemplate']]
    # Write to a new CSV file
    new_df.to_csv(output_file, index=False)

def csv_to_json(input_file, output_file):
    df = pd.read_csv(input_file)

    json_data = []
    num_rows = df.shape[0]
    window_size = 20
    for start in range(0, num_rows, window_size):
        end = min(start + window_size, num_rows)
        window_df = df.iloc[start:end]

        # Splice EventTemplate and determine output
        input_data = ','.join(window_df['EventTemplate'])
        if 'abnormal' in window_df['Label'].values:
            output_data = 'abnormal'
        else:
            output_data = 'normal'

        # Build a JSON object and add it to the list
        json_obj = {
            "instruction": "Analyze a series of log entries and determine whether the overall status is 'normal' or 'abnormal'. Your response should follow the following format: x-y, where x represents the status of the log sequence (you can answer only one of 'normal' or 'abnormal') and y provides a concise explanation of why the status is normal or abnormal.",
            "input": input_data,
            "output": output_data
        }
        json_data.append(json_obj)

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)
def write_10000_elements(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file:
        # Read the JSON data from the input file
        data = json.load(input_file)
    # print(data[40000])
    # Extract the last 10,000 elements
    last_10000_elements = data[40000:]

    with open(output_file_path, 'w') as output_file:
        # Write the last 10,000 elements into the output file
        json.dump(last_10000_elements, output_file, indent=4)

def split(file_path):
    # file_path = 'explantion_20bgl_prompt_prase.json'

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    df = pd.DataFrame(data)

    # Although this division is 8:2, we only use the first 40,000 log sequences of the training set for training, and the rest are the test set
    # Divide the data set into training set and test set, 80% training set, 20% test set
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)


    print("The number of training set data:", len(train_data))
    print("The number of test set data:", len(test_data))


    with open('train_bgl.json', 'w', encoding='utf-8') as file:
        json.dump(train_data.to_dict(orient='records'), file, ensure_ascii=False, indent=4)

    with open('test_bgl.json', 'w', encoding='utf-8') as file:
        json.dump(test_data.to_dict(orient='records'), file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    extract_columns('BGL.log_structured.csv','bgl.csv')
    csv_to_json('bgl.csv','modified_bgl.json')
    split('modified_bgl.json')
    # test_bgl.json and the following files are used for testing
    write_10000_elements('train_bgl.json','modified_bgl_after_4w.json')
