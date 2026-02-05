# tạo list
# list_rong = []
# list_snt = [2, 4, 5, 6, 8, 10, 12, 15, 18]
# list_dongvat = ["heo", "chó", "bò", "gà"]

# list_dongvat.append("vịt")
# print(list_dongvat)

# in theo vị trí
# print(list_dongvat[3])
# print(list_snt[-2])

# Độ dài list
# print(len(list_dongvat))

# duyệt list có 2 các 
# cách 1 duyệt theo số thứ tự
# for i in range(len(list_snt)):
#     print(list_snt[i])

# cách 2 
# for x in list_dongvat:
#     print(x)


# thêm phần tử vào list
# list_dongvat.append("vịt") # thêm vào cuối
# print(list_dongvat)

# list_dongvat.insert(2, "mèo") # thêm vào vị trí số 2
# print(list_dongvat)

# xóa phần tử khỏi list remove với pop
# list_dongvat.pop(3)
# print(list_dongvat)

# list_dongvat.remove("chó")
# print(list_dongvat)

# sửa
# list_dongvat[0] = "mèo"
# print(list_dongvat)

# kiểm tra
# if "vịt" in list_dongvat:
#     print("có vịt")
# else:
#     print("không có vịt")


# x = list_snt.count(1)
# print(x)


# list_snt.sort()
# print(list_snt)

# list_dongvat.reverse()
# print(list_dongvat)

# sln = sum(list_snt)
# print(sln)



n=int(input('Nhập số: '))
list_rong=[]
for i in range(0,4):
    n=int(input('Nhập số: '))
    list_rong.append(n)
print('Danh sách: ', list_rong)
max=max(list_rong)
min=min(list_rong)
print('=> Max: ',max,', Min: ',min)