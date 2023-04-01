#! /usr/bin/env python3
import sys
import re

    
for line in sys.stdin:
    try:
        thread_lst = line.strip().split("/")[0].split(" ")
        thread = thread_lst[1] + " " + thread_lst[2]
        thread = thread.strip("[")
        print("%s\t%d" % (thread, 1))
    except ValueError as e:
        print(e)
        continue
