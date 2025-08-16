import os

global basepath

basepath = os.path.split(__file__)[0]

print(basepath)
