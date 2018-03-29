from collections import defaultdict
import pickle
import random
# emit = defaultdict(int)
# transition = defaultdict(int)
# context = defaultdict(int)


def HMM_VITERBI(w, X):
    with open("../../data/wiki-en-train.norm_pos") as text:
        possible_tags["</s>"] = 1
        for line in text:
            words = line.split()
            words.append("</s>")
            l = len(words)
            best_score = {}
            best_edge = {}
            best_score["0 <s>"] = 0
            best_edge["0 <s>"] = "NULL"
            for i in range(l):
                for prev in possible_tags.keys():
                #最初にくる品詞をまわす
                    for next_tag in possible_tags.keys():
                    #その次にくる品詞をまわす
                    #nextは関数にもあるので？、名前として使えない（typeと同様）
                        #print(prev, next_tag)
                        if str(i) + " " + prev in best_score: #and prev + " " + next_tag in dict_transition:
                            # print(transition[prev + " " + next_tag])
                            # print(emission[next_tag + " " + words[i]])
                            score = best_score[str(i) + " " + prev] + PREDICT(w, CREATE_TRANS(prev, next_tag)) + PREDICT(w, CREATE_EMIT(next_tag, word[i]))
                            if (str(i+1) + " " + next_tag not in best_score) or (best_score[str(i+1) + " " + next_tag] < score):
                                best_score[str(i+1) + " " + next_tag] = score
                                best_edge[str(i+1) + " " + next_tag] = str(i) + " " + prev

        #文末処理
        print(best_edge)
        for prev in possible_tags.keys():
            transition = "T {} </s>".format(prev)
            emission = "E </s> </s>"
            if (str(l) + " " + prev) in best_score and transition in dict_transition:
                score = best_score[str(l) + " " +prev] + weight[transition] + weight[emission]
                if (str(l+1) + "</s>") not in best_score or best_score[str(l+1) + "</s>" ] < score:
                    best_score[str(l+1), "</s>"] = score
                    best_edge[str(l+1), "</s>"] = "{} {}".format(i, prev)
                    print(best_edge[str(l+1), "</s>"])



        #後ろ向きステップ
        tags = []

        next_edge = best_edge[str(l+1) + " " + "</s>"]
        while next_edge != "0 <s>":
            position, tag = next_edge.split()
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        return tags
        # Y_hatのこと



def PREDICT(w, phi):
    score = 0
    for name, value in phi.items():
        score += value * w[name]
    return score


def CREATE_FEATURES(X, Y):
    phi = dict()
    for i in range(len(Y) + 1):
        if i == 0:
            first_tag = "<s>"
        else:
            first_tag = Y[i-1]
        if i == len(Y):
            next_tag = "</s>"
        else:
            next_tag = Y[i]
        phi.update(CREATE_TRANS(first_tag, next_tag))
    for i in range(len(Y)):
        phi.update(CREATE_EMIT(Y[i], X[i]))
    return phi


def CREATE_EMIT(Y, X):
    phi = defaultdict(int)
    phi["E {} {}".format(Y, X)] += 1
    return phi


def CREATE_TRANS(first_tag, next_tag):
    phi = defaultdict(int)
    phi["T {} {}".format(first_tag, next_tag)] += 1
    return phi



if __name__ == '__main__':
    weight = defaultdict(int)
    epoch = 1
    data = []
    possible_tags = dict()
    dict_t_e = defaultdict(int)
    dict_transition = defaultdict(int)
    with open("../../data/wiki-en-train.norm_pos") as input_file:
        for line in input_file:
            temp_list = list()
            word_tags = line.strip().split()
            previous = "<s>"
            X = list()
            Y_prime = list()
            for word_tag in word_tags:
                word, tag = word_tag.split("_")
                possible_tags[tag] = 1 #出てきたやつ
                transe = "T {} {}".format(previous, tag)
                dict_transition[transe] = 1
                previous = tag #置き換える
                X.append(word)
                Y_prime.append(tag)
            dict_transition["T {} </s>".format(tag)] = 1
#            X.append('</s>')
#            Y_prime.append('</s>')
            temp_list.append((X, Y_prime))
            data.append(temp_list)
        possible_tags["<s>"] = 1
        possible_tags["</s>"] = 1
    dict_transition = dict(dict_transition)

    for j in range(epoch):
        print("epoch {}".format(j)) # エポックいくつか
        random.shuffle(data)
        for i, line in enumerate(data):
            if i % 100 == 0:
                print(i)
            for x, y_prime in line:
#                print(x, y_prime)
                y_hat = HMM_VITERBI(weight, x)
                phi_prime = CREATE_FEATURES(x, y_prime)
                phi_hat = CREATE_FEATURES(x, y_hat)
                for key, value in phi_prime.items():
                    weight[key] += value
                for key, value in phi_hat.items():
                    weight[key] -= value
    with open("weight_file.dump", "wb") as output_file:
        pickle.dump((dict(weight), dict_transition, possible_tags), output_file)



#
# # ここからチュートリアル５（HMM）
#
# with open("../../data/wiki-en-train.norm_pos") as text:
#     #テストファイルなら ../../test/05-train-input.txt
#     #テストファイルで実行中。あとで戻すのを忘れずに。（もどした）
#     #擬似コードの「＋＋」が「＋＝１」なのかよくわかってないのであとで確認する
#     #あってた
#     for line in text:
#         previous = "<s>"
#         context[previous] += 1
#         wordtags = line.strip().split(" ")
#         for wordtag in wordtags:
#             word, tag = wordtag.split("_")
#             #splitされたものがwordとtagにそれぞれ入る
#             #natural_JJがword = naturalとtag = JJみたいな感じで
#             transition[previous + " " + tag] += 1 #遷移
#             context[tag] += 1 #文脈
#             emit[tag + " " + word] += 1 #生成
#             previous = tag
#             #前の単語の種類が何だったかを更新してるっぽい
#         transition[previous + " </s>"] += 1
#
# # print(emit)
# # print(transition)
# # print(context)
#
#
# with open("model_file.word", "w") as text:
#     for key, value in transition.items():
#         previous, word = key.split()
#         value = value/context[previous]
#         text.write("T" + " " + key + " " + str(value) + "\n")
#     for key, value in emit.items():
#         previous, word = key.split()
#         value = value/context[tag]
#         text.write("E" + " " + key + " " + str(value) + "\n")
#
# #model_fileがなんかずれるので質問する
