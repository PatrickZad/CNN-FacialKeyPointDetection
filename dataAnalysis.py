import pandas as pd
df=pd.read_csv('./training.csv')
booldf=df.isnull()
print(booldf.sum())
numeye=0
numbrow=0
nummouth=0
for i in range(7049):
    row=booldf.iloc[i]
    if not row.left_eye_inner_corner_x and \
        not row.left_eye_outer_corner_x and \
        not row.right_eye_inner_corner_x and \
        not row.right_eye_outer_corner_x :#全部标记
        numeye+=1
    if  not row.left_eyebrow_inner_end_x and \
        not row.left_eyebrow_outer_end_x and \
        not row.right_eyebrow_inner_end_x and \
        not row.right_eyebrow_outer_end_x:#全部标记
        numbrow+=1
    if not row.mouth_left_corner_x and \
       not row.mouth_right_corner_x and \
        not row.mouth_center_top_lip_x :#全部标记
        nummouth+=1
print(numeye)#2247
print(numbrow)#2190
print(nummouth)#2260