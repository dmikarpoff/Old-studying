#! /usr/bin/python

import os
import random

prefix = "../lfw/"

if __name__ == "__main__":
    csv = ""
    csv_test = ""
    ld = os.listdir(prefix)
    out = open('csv.txt', 'w')
    out_test = open('csv.test.txt', 'w')
    for fname in ld:
        if (fname.endswith(".lbl")):
            f = open(prefix + fname, 'r')
            gender = int(f.read())
            person = fname[0:len(fname) - 4]
            path = prefix + person + "/"
            phs = os.listdir(path)
            for i in phs:
                if not i.endswith(".lbl"):
                    r = random.randint(0, 9)
                    csv = csv + path + i + ';' + str(gender) + '\n'
                    if (r % 10 == 0):
                        csv_test = csv_test + path + i + ';' + str(gender) + '\n'
    out.write(csv)
    out_test.write(csv_test)                
