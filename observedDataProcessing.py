import pandas as pd
import os
import numpy as np
import csv

# 각 파일의 단위당 데이터 범위
START = 7
END = 101

# 도수분포표 범위
BINS = 250

re_list = []  # 날짜별 r_edep 리스트
for i in range(31):
    re_list.append([])
hist_list = []  # 날짜별 도수분포표 리스트
for i in range(31):
    hist_list.append([])

dir_path = 'D:/observed_data/raw_data1'  # 디렉토리 경로

file_list = os.listdir(dir_path)  # 디렉토리에 있는 파일 이름 리스트
file_list.reverse()  # 리스트 역순
print(file_list)
file_num = len(file_list)

date_list = []

for i in range(file_num):
    file_name = file_list.pop()  # 뒤에서부터(오래된 파일부터)
    file_path = dir_path + '/' + file_name

    date = file_name[6:12]
    if date in date_list:
        pass
    else:
        date_list.append(date)

    # txt 파일을 읽어서 Ch1, Ch2에 해당하는 데이터를 dataframe에 저장
    df = pd.read_csv(file_path,
                     sep='\s+',
                     engine='python',
                     usecols=[2, 3],
                     names=['Ch1', 'Ch2'],
                     header=None)
    dataset_num = int(len(df['Ch1'])/102)

    for j in range(1, dataset_num):
        # 각 단위의 데이터 범위
        start = (j-1) * 102 + START
        end = (j-1) * 102 + END

        Ch1_list = []
        Ch2_list = []
        for k in range(start, end+1):
            Ch1_list.append(df.loc[k, 'Ch1'])
            Ch2_list.append(df.loc[k, 'Ch2'])

        # 1023에서 빼기
        for k in range(END-START+1):
            Ch1_list[k] = 1023 - Ch1_list[k]
            Ch2_list[k] = 1023 - Ch2_list[k]

        # 데이터 누적
        sum_Ch1 = sum(Ch1_list)
        sum_Ch2 = sum(Ch2_list)

        # r_edep 계산 및 날짜별 r_edep 리스트 추가
        re = sum_Ch1/sum_Ch2*100
        re_list[len(date_list)-1].append(re)
    print(date_list)

# 도수분포표 그리기
date_num = len(date_list)
date_list.reverse()
for i in range(date_num):
    bins = np.arange(0, BINS, 1)
    hist, bins = np.histogram(re_list[i], bins)
    hist = hist.tolist()
    hist_list[i].append(date_list.pop())
    hist_list[i].append(hist)

# 도수분포표를 csv 파일로 저장하기
with open('hist1.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerows(hist_list)
