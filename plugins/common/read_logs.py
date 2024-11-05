import pandas as pd

def read_log(**kwargs):
    print(f"pandas version: \n{pd.__version__}\n\n")
    
    df = pd.read_csv(filepath_or_buffer="data/logs/log-241001.txt", delimiter=" ")
    print(f"df: \n{df}\n\n")
    
    df_json = df.to_json(orient="split")
    
    return df_json