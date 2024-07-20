import os
top = './datasets/'
a =  os.walk(top, topdown=False)
for root, dirs, files in a:
    # for i in range(50):
        # print(i)
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))
# for i in range(50):
    # print(i)
print('end')