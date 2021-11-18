# import os
import xlrd
import xlwt
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pylab as pl
import scipy.signal as signal

data = xlrd.open_workbook('F:/xxx.xlsx')
table = data.sheets()[0]
nrows = table.nrows
ncols = table.ncols

x = np.array(table.row_values(0)) # [0:50]

plt.figure(figsize=(16,4))
plt.plot(np.arange(len(x)),x)

print(x[signal.argrelextrema(x, np.greater_equal)])
print(signal.argrelextrema(x, np.greater_equal))

print(x[signal.argrelextrema(x, np.less)])
print(signal.argrelextrema(x, np.less))

workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('My Worksheet')

arr0 = signal.argrelextrema(x, np.less)
list0 =x[signal.argrelextrema(x, np.less)]
arr1 = signal.argrelextrema(x, np.greater_equal)
list1 =x[signal.argrelextrema(x, np.greater_equal)]


worksheet.write(0, 0, 'Max P')
worksheet.write(1, 0, 'Max A')
worksheet.write(3, 0, 'MIN P')
worksheet.write(4, 0, 'MIN A')

for i in range(len(list1)):
        print(arr1[0][i])
        print(list1[i])
        worksheet.write(1, i+1, int(str(arr1[0][i])))
        worksheet.write(0, i+1, float(str(list1[i])))

for i in range(len(list0)):
    print(arr0[0][i])
    print(list0[i])
    worksheet.write(4, i+1, int(str( arr0[0][i])))
    worksheet.write(3, i+1, float(str( list0[i])))


workbook.save('D:/xxx.xls')
plt.plot(signal.argrelextrema(x,np.greater_equal)[0],x[signal.argrelextrema(x, np.greater_equal)],'o')
plt.plot(signal.argrelextrema(x,np.less)[0],x[signal.argrelextrema(x, np.less)],'+')
# plt.plot(peakutils.index(-x),x[peakutils.index(-x)],'*')
plt.show()

