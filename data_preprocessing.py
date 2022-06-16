import pandas as pd


df = pd.read_csv('dataset/data.csv', encoding='euc-kr')
#'euc-kr' 인코딩을 사용해 한글 csv 파일 읽기

df.rename(columns={'내국인업종코드(SB_UPJONG_CD)': 'SB_UPJONG_CD'}, inplace=True)
df.rename(columns={'성별(SEX_CCD)': 'SEX_CCD'}, inplace=True)
#df.rename(columns={'연령대별(AGE_GB)': 'AGE_GB'}, inplace=True)
df.rename(columns={'카드이용건수(USECT_CORR)': 'USECT_CORR'}, inplace=True)

df['SB_UPJONG_CD'] = df['SB_UPJONG_CD'].str.replace(r'\D', '')

df['SEX_CCD'] = df['SEX_CCD'].str.replace('M', '1')
#SEX_CCD의 M을 1로 변환

df['SEX_CCD'] = df['SEX_CCD'].str.replace('F', '0')
#SEX_CCD의 F를 0으로 변환

data = df.dropna(how='any')

data = data.reset_index(drop=True)



data.to_csv('dataset/processed_data.csv', encoding='euc-kr', index = False)



cd = pd.read_csv('dataset/code.csv', encoding='euc-kr')

cd.rename(columns={'대분류(SB_L_UPJONG_NM)': 'SB_L_UPJONG_NM'}, inplace=True)
cd.rename(columns={'중분류(SB_M_UPJONG_NM)': 'SB_M_UPJONG_NM'}, inplace=True)
cd.rename(columns={'내국인업종분류(SB_UPJONG_NM)': 'SB_UPJONG_NM'}, inplace=True)
cd.rename(columns={'내국인업종코드(SB_UPJONG_CD)': 'SB_UPJONG_CD'}, inplace=True)

cd.to_csv('dataset/processed_code.csv', encoding='euc-kr', index = False)