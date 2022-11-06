import pandas as pd
import os

# 각 파일의 단위당 데이터 범위
START = 7
END = 101

# 년월
YEAR = 2020
for MONTH in range(1, 8):
    date_list = [0 for i in range(31)]
    Ch1_list = [0 for i in range(31)]
    Ch2_list = [0 for i in range(31)]
    re_list = [0 for i in range(31)]
    dataset_list = [0 for i in range(31)]

    dir_path = 'D:/observedData/rawData/'+str(YEAR)+'.'+str(MONTH)  # 디렉토리 경로

    file_list = os.listdir(dir_path)  # 디렉토리에 있는 파일 이름 리스트
    file_list.reverse()  # 리스트 역순
    print(file_list)
    file_num = len(file_list)

    for i in range(file_num):
        file_name = file_list.pop()  # 뒤에서부터(오래된 파일부터)
        file_path = dir_path + '/' + file_name
        if os.path.getsize(file_path) == 0:
            print('file is empty')
            continue
            
        print(file_path)

        day = int(file_name[10:12])
        date_list[day-1] = str(YEAR)+str(MONTH)+str(day)
        
        # txt 파일을 읽어서 Ch1, Ch2에 해당하는 데이터를 dataframe에 저장
        try:  # 에러 처리
            df = pd.read_csv(file_path,
                            sep='\s+',
                            engine='python',
                            usecols=[2, 3],
                            names=['Ch1', 'Ch2'],
                            header=None)

        except Exception as e:
            print(e) 
            continue
        
        dataset_num = int(len(df['Ch1'])/102)
        dataset_list[day-1] += dataset_num

        for j in range(1, dataset_num):
            # 각 단위의 데이터 범위
            start = (j-1) * 102 + START
            end = (j-1) * 102 + END

            Ch1 = []
            Ch2 = []
            for k in range(start, end+1):
                Ch1.append(df.loc[k, 'Ch1'])
                Ch2.append(df.loc[k, 'Ch2'])

            # 1023에서 뺴고 누적
            sum_Ch1 = 1023*(END-START+1)-sum(Ch1)
            sum_Ch2 = 1023*(END-START+1)-sum(Ch2)
            Ch1_list[day-1] += sum_Ch1
            Ch2_list[day-1] += sum_Ch2

            re = sum_Ch1 / sum_Ch2 * 100
            re_list[day-1] += re

    for i in range(31):
        if dataset_list[i] == 0:
            pass

        else:
            Ch1_list[i] /= dataset_list[i]
            Ch2_list[i] /= dataset_list[i]
            re_list[i] /= dataset_list[i]


    raw_data = {'date': date_list,
                'Ch1': Ch1_list,
                'Ch2': Ch2_list,
                'r_edep': re_list
                }

    df = pd.DataFrame(raw_data)

    df.to_csv(str(YEAR)+'.'+str(MONTH)+'.csv', sep=',', na_rep='NaN')
 
