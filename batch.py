import sys
import json

json_name = sys.argv[1]

try:
    file_handle = f = open(json_name)
    json_data = json.load(file_handle)

    print("json data : ", json_data)
    f.close()
except FileNotFoundError:
    print("Wrong file or file path : ", json_name)
