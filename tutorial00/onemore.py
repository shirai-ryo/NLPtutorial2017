my_dict = {}

with open("../../data/wiki-en-train.word", "r") as text:
    for line in text:
        words = line.split()
        for word in words:
            if word in my_dict:
                my_dict[word] += 1
            else:
                my_dict[word] = 1

    print(my_dict.items())
