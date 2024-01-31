import os

folders = next(os.walk('Data'))[1]

print(folders)

if 'mat' in folders:
    folders.pop(folders.index('mat'))
    
print(len(folders))