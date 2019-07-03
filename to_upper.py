import pandas as pd

# 小写变大写
if __name__ == '__main__':
    tmp_list = []
    raw_data = pd.read_csv('data/RELIGION_TMP_.csv', header=None)
    for tmp_data in raw_data[0].values:
        tmp_list.append(tmp_data.upper())
    new_df = pd.DataFrame({
        'la': tmp_list
    })
    new_df.to_csv('data/RELIGION.csv', index=False, header=None)