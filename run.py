# coding=utf-8
import cv2
import pytesseract
import os
import textdistance
import numpy as np
import pandas as pd

ROOT_PATH = os.getcwd()
IMAGE_PATH = os.path.join(ROOT_PATH, 'images/ktp.png')
LINE_REC_PATH = os.path.join(ROOT_PATH, 'data/ID_CARD_KEYWORD.csv')
NEED_COLON = [3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21]
NEXT_LINE = 9
ID_NUMBER = 3


# 对身份证上的信息进行初步OCR
def ocr_raw(image_path):
    # (1) Read
    img_raw = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    # (2) Threshold
    th, threshed = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
    # (3) Detect
    result_raw = pytesseract.image_to_string(threshed, lang="ind")
    return result_raw


# 去除result_raw中的无效信息
def strip_op(result_raw):
    result_list = result_raw.split('\n')
    new_result_list = []
    for tmp_result in result_list:
        if tmp_result.strip(' '):
            new_result_list.append(tmp_result)
    return new_result_list


def main():
    raw_df = pd.read_csv(LINE_REC_PATH, header=None)
    result_raw = ocr_raw(IMAGE_PATH)
    result_list = strip_op(result_raw)
    loc2index = dict()
    for i, tmp_line in enumerate(result_list):
        for j, tmp_word in enumerate(tmp_line.split(' ')):
            tmp_sim_list = [textdistance.damerau_levenshtein.normalized_similarity(tmp_word_, tmp_word.strip(':')) for tmp_word_ in
                            raw_df[0].values]
            tmp_sim_np = np.asarray(tmp_sim_list)
            arg_max = np.argmax(tmp_sim_np)
            if tmp_sim_np[arg_max] >= 0.6:
                loc2index[(i, j)] = arg_max

    print('--------processed------------')
    last_result_list = []
    useful_info = False
    for i, tmp_line in enumerate(result_list):
        tmp_list = []
        for j, tmp_word in enumerate(tmp_line.split(' ')):
            tmp_word = tmp_word.strip(':')
            if (i, j) in loc2index:
                useful_info = True
                if loc2index[(i, j)] == NEXT_LINE:
                    last_result_list.append(tmp_list)
                    tmp_list = []
                tmp_list.append(raw_df[0].values[loc2index[(i, j)]])
                if loc2index[(i, j)] in NEED_COLON:
                    tmp_list.append(':')
            elif tmp_word == ':' or tmp_word == '':
                continue
            else:
                tmp_list.append(tmp_word)
        if useful_info:
            if len(last_result_list) > 2 and ':' not in tmp_list:
                last_result_list[-1].extend(tmp_list)
            else:
                last_result_list.append(tmp_list)

    for tmp_data in last_result_list:
        if 'NIK' in tmp_data:
            print(''.join(tmp_data[2:]))
            id_number = ''.join(tmp_data[2:])
            if "D" in id_number:
                id_number = id_number.replace("D", "0")
            if "?" in id_number:
                id_number = id_number.replace("?", "7")
            if "L" in id_number:
                id_number = id_number.replace("L", "1")
            while len(tmp_data) > 2:
                tmp_data.pop()
            tmp_data.append(id_number)
            break

    for tmp_data in last_result_list:
        print(tmp_data)


if __name__ == '__main__':
    main()
