#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import subprocess
import sys
import argparse

py_version = sys.version
print(py_version)
if py_version[:4] == '3.9' or py_version[:4] == '3.10' or py_version[:4] == '3.11':
    py_requires = 'python' + sys.version[:4]
else:
    py_requires = 'python3.8'
print(py_requires)

file = os.getcwd() 


#subprocess.run(["cd",file]), check=True, stdout=subprocess.PIPE).stdout
os.system('cd ' + file)

os.system('./make_version.sh')

print("version file updated")
print('*'*100)

## git commit - with message added in ./make_version.sh
# if args.message:
#     subprocess.run(["git", "add", "."], check=True, stdout=subprocess.PIPE).stdout
#     subprocess.run(["git", "commit", "-am", args.message], check=True, stdout=subprocess.PIPE).stdout
#     print('git commit done with message: ' + args.message)
# # print('git commit done')

subprocess.run(["git", "pull"], check=True, stdout=subprocess.PIPE).stdout
print('git pull done')
print('*'*100)

subprocess.run(["git", "push"], check=True, stdout=subprocess.PIPE).stdout
print('*'*100)
print('removing dist and build folders')

if os.path.exists(file+'/dist'):
    os.system('sudo rm -rf '+file+'/dist')
    os.system('sudo rm -rf '+file+'/build')
#subprocess.run(["ls"]),check=True, stdout=subprocess.PIPE).stdout
os.system("ls")

os.system(py_requires + ' -m build')

print('*'*100)
print('wheel built')
print(py_requires + ' -m pip install '+file + '/dist/' +os.listdir(file +'/dist')[-1] + ' --break-system-packages')
os.system(py_requires + ' -m pip install '+file + '/dist/' +os.listdir(file +'/dist')[-1] + ' --break-system-packages')

print('package installed')
print('*'*100)
os.system(py_requires + ' -m twine upload dist/*')