#! /usr/bin/python

import random
import os

filename = "data.txt"
path = "/home/javier/Studying/MachineLearning/lfw/"
amount = 60000
sep = ';'

if __name__ == "__main__":
    f = open("data.txt", "w");
    name_list = os.listdir(path)
    output = ""
    for i in range(amount):
        same = random.randint(0, 1)
        if same:
            person = random.randint(0, len(name_list) - 1)
            photo_list = os.listdir(path + name_list[person] + "/")
            im1 = random.randint(0, len(photo_list) - 1)
            im2 = random.randint(0, len(photo_list) - 1)
            output = output + path + name_list[person] + "/" + photo_list[im1] + sep
            output = output + path + name_list[person] + "/" + photo_list[im2] + sep
        else:
            p1 = random.randint(0, len(name_list) - 1)
            p2 = random.randint(0, len(name_list) - 1)
            while p2 == p1:
                p2 = random.randint(0, len(name_list) - 1)
            ph_l1 = os.listdir(path + name_list[p1] + "/")
            ph_l2 = os.listdir(path + name_list[p2] + "/")    
            im1 = random.randint(0, len(ph_l1) - 1)
            im2 = random.randint(0, len(ph_l2) - 1)
            output = output + path + name_list[p1] + "/" + ph_l1[im1] + sep
            output = output + path + name_list[p2] + "/" + ph_l2[im2] + sep
        output = output + str(same) + "\n"
        if i % 100 == 0:
            print i
    f.write(output)        
			    
