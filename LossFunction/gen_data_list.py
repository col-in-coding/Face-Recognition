import os

if __name__ == "__main__":
    folder = "data"
    dictionary = {}
    with open("data_list.txt", 'a') as f:
        for filename in os.listdir(folder):
            labelname = filename.split('_')[0]
            if labelname not in dictionary.keys():
                label = len(dictionary)
                dictionary[labelname] = label
            f.write(f"{filename} {label}" + os.linesep)
