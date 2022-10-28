import csv
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# col_list = ["Close","KL"]
# data = pd.read_csv('vn2022.csv',usecols = col_list)
# # print(data_import.values.shape)
# #
# # arr_str =data_import.values.flatten()
# # print(arr_str.shape)
# # print(type(data))
# diem = np.asarray(data.loc[:,"Close"])# lay ra row dau
# # print(type(diem))
# diem =diem[::-1]/1600
# kl  = np.asarray(data.loc[:,"KL"])
# kl = kl[::-1]
#
# data1 = pd.read_csv('vn2008.csv',usecols = col_list)
# diem1 = np.asarray(data1.loc[:,"Close"])# lay ra row dau
# # print(type(diem))
# diem1 =diem1[::-1]/1200
# kl1  = np.asarray(data1.loc[:,"KL"])
# kl1 = kl1[::-1]
# fir, ax = plt.subplots(2, 2)
# ax[0,0].plot(diem)
# ax[1,0].plot(kl)
# ax[0,1].plot(diem1)
# ax[1,1].plot(kl1)
# plt.show()
############################################
# # van draw tat ca du lieu nhung truc x chi hien thi cac thang
# col_list = ["Date","Close","KL"]
# data = pd.read_csv('vn.csv',usecols = col_list)
# date = diem = np.asarray(data.loc[:,"Date"])
# date = date[::-1]
# diem = np.asarray(data.loc[:,"Close"])# lay ra row dau
# # print(type(diem))
# diem =diem[::-1]/1600
# kl  = np.asarray(data.loc[:,"KL"])
# kl = kl[::-1]
# old_month = ""
# month=[]
# # index =[]
# i =0
# for s in date:
#     s2 = s[s.index("/") + 1:]
#     if s2[0]=='0':
#         s2 =s2[1:]
#     if s2!=old_month:
#         month.append(s2)
#         # index.append(i)
#         old_month =s2
#         # print(s2)
#     # else:
#     #     month.append("")
#     i+=1
# month = np.asarray(month)
# print(month.shape[0])
# fig,ax = plt.subplots(1)
# plt.xticks(np.linspace(0,i,month.shape[0]), month) # hien trong dai kia
# ax.plot(diem)
# plt.show()
# #
# # plt.plot(date,diem)
# # plt.show()
#######################################3
col_list = ["Close","KL"]
data = pd.read_csv('vn.csv',usecols = col_list)
# print(data.Close)
########## dao nguoc data frame #############
# data = data.iloc[::-1]
# data = data.reindex(index=data.index[::-1])
# print(data)
################### lay ra diem ##########################

# diem = np.asarray(data.Close[:300])
diem = data.Close[:300]
# dao nguoc cac chi so cua hang de ve do thi
diem = diem.loc[::-1].reset_index(drop=True)
# diem = diem[::-1]
# kl = np.asarray(data.KL[:300])
kl =data.KL[:300]
kl = kl.loc[::-1].reset_index(drop=True)
# kl =kl[::-1]
print(diem)
print(kl)
# print(diem)
# diem = diem[:200]
# print(diem)
###################tinh trung binh tu dau#############################
# Convert array of integers to pandas series
numbers_series = pd.Series(diem)
# Get the window of series of
# observations till the current time
windows = numbers_series.expanding()
# Create a series of moving averages of each window
moving_averages = windows.mean()
# Convert pandas series back to list
moving_averages_list = moving_averages.tolist()
################### tinh trung binh tu n phan tu #############
mean50 = diem.rolling(10).mean()# trung binh theo kieu cuon chieu 1000 phan tu
mean100  = diem.rolling(30).mean()
klmean50 =  kl.rolling(10).mean()
klmean100 =  kl.rolling(30).mean()
# print(mean)
fig, (ax,ax1) = plt.subplots(2)
ax.plot(diem,label = 'diem')
ax.plot(mean50,label ='10')
ax.plot(mean100,label ='30')
# plt.plot(moving_averages_list)
ax.legend()

ax1.plot(kl,label = 'kl')
ax1.plot(klmean50,label ='10')
ax1.plot(klmean100,label ='30')
# plt.plot(moving_averages_list)
ax1.legend()
plt.show()