def andop(a,b):
    l=len(a)
    c=[]
    for i in range(l):
        if(a[i]==1 and b[i]==1):
            c.append(1)
        else:
            c.append(0)
    return c

def orop(a,b):
    l=len(a)
    c=[]
    for i in range(l):
        if(a[i]==0 and b[i]==0):
            c.append(0)
        else:
            c.append(1)
    return c

num1=[1,0,1,1]
num2=[0,1,1,0]
print("Result from andop ", andop(num1,num2))
print("Result from orop ",orop(num1,num2))
