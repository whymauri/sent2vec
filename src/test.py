import sent2vec
model = sent2vec.Sent2vecModel()
model.start_nnsent("../testdata.txt")
nnsent = model.single_nn(10,"../testdata.txt","how do i set up wordpress")
sentences = [sentence.split(" ",2) for sentence in nnsent]
res = {i:{"dis":sentences[i][0],"index":sentences[i][1],"sentence":sentences[i][2]} for i in range(len(sentences))}