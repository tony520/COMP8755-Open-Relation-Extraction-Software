def transferSeq(sents, tags):
    res = ""
    argdic = {"A0":[], "A1":[], "A2":[], "A3":[], "A4":[], "A5":[], "P-":[]}
    assert len(sents) == len(tags)
    for i in range(len(tags)):
        key = tags[i][0:2]
        if key in argdic:
            argdic[key].append(sents[i])

        if tags[i][0] == 'A':
            sub = '<arg' + tags[i][1] + '> '
            if sub not in res:
                res = res + ' ' + sub + ' ' + sents[i]
            else:
                res = res + ' ' + sents[i]
        elif tags[i][0] == 'P':
            sub = ' <rel> '
            res = res + ' ' + sub + ' ' + sents[i]
        else:
            continue
    
    arr = res.split()
    transferStr = ""
    
    i = 0
    while i < len(arr):
        if '<arg0>' in arr[i]:
            dis = len(argdic['A0'])
            temp_s = ' '.join([w for w in argdic['A0']])
            transferStr = transferStr + ' <arg0> ' + temp_s + ' </arg0> '
            i += dis
        elif '<arg1>' in arr[i]:
            dis = len(argdic['A1'])
            temp_s = ' '.join([w for w in argdic['A1']])
            transferStr = transferStr + ' <arg1> ' + temp_s + ' </arg1> '
            i += dis
        elif '<arg2>' in arr[i]:
            dis = len(argdic['A2'])
            temp_s = ' '.join([w for w in argdic['A2']])
            transferStr = transferStr + ' <arg2> ' + temp_s + ' </arg2> '
            i += dis
        elif '<arg3>' in arr[i]:
            dis = len(argdic['A3'])
            temp_s = ' '.join([w for w in argdic['A3']])
            transferStr = transferStr + ' <arg3> ' + temp_s + ' </arg3> '
            i += dis
        elif '<arg4>' in arr[i]:
            dis = len(argdic['A4'])
            temp_s = ' '.join([w for w in argdic['A4']])
            transferStr = transferStr + ' <arg4> ' + temp_s + ' </arg4> '
            i += dis
        elif '<arg5>' in arr[i]:
            dis = len(argdic['A5'])
            temp_s = ' '.join([w for w in argdic['A5']])
            transferStr = transferStr + ' <arg5> ' + temp_s + ' </arg5> '
            i += dis
        elif '<rel>' in arr[i]:
            dis = len(argdic['P-'])
            temp_s = ' '.join([w for w in argdic['P-']])
            transferStr = transferStr + ' <rel> ' + temp_s + ' </rel> '
            i += dis
        else:
            i += 1
    return transferStr


# Get all transferred sequences
def getTripleSeq(data):    
    triples = []
    for i in range(len(data)):
        ts = transferSeq(data[i][0], data[i][1])
        triples.append(ts)
    
    return triples

def gen_train_seq(data, filename):
    tps = getTripleSeq(data)
    with open(filename, 'w') as f:
        for line in tps:
            f.write(line)
            f.write('\n')

gen_train_seq(training_data, 'tgt-src-gs-new.txt')
gen_train_seq(testing_data, 'tgt-val-gs-new.txt')