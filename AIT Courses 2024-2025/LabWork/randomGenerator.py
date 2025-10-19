import random

arr = []
numElements = int(input())
startNum=int(input())
endNum=int(input())

for i in range(numElements):
    x = random.randint(startNum, endNum)
    arr.append(x)
    print(x)

sum1 = 0
sum2 = 0
for i in range(numElements):
    #print("i and arr[i]",i,arr[i])
    sum1 = sum1 + arr[i]
print("Sum1=",sum1)

for i in arr:
    sum2 = sum2 + i
    print(i)
#print("Sum2=",sum2)

min = arr[0]
for i in arr:
    if(i < min):
        min = i
print("Minimum value is ",min)

a = 0
sum = 0 
for i in arr:
    a += 1
    sum += i
print("middle value = ",sum/a)

print("average is", sum2/numElements)






