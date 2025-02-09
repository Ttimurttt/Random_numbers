import numpy as np 

def make_ngrams(text, order):
    """ Divides a text in "ngrams" \n
    For example "Timur" in order 3 is 
    "Tim", "imu", "mur" """
    
    ngrams = []

    ngrams_count = len(text)-order+1
    
    count = np.ones(ngrams_count)
    
    for num in range(ngrams_count): # loops through every ngram
        ngram = text[num:num+order] # gets ngram
        
        if (ngram in ngrams):
            first_ind = ngrams.index(ngram)
            count[first_ind] += 1
        ngrams.append(ngram) # adds it to a list of ngrams
        
    return ngrams, count

def What_comes_after(text, ngrams, count, order):
    after = []
    for x in range(len(count)):
        to_append = []
        temp_ngrams = list(np.copy(ngrams))
        for n in range(int(count[x])):
            ind = temp_ngrams.index(ngrams[x])
            try: 
                to_append.append(text[ind+order+n])
            except IndexError:
                if len(to_append) == 0:
                    to_append.append('@')
            temp_ngrams.pop(ind)
        after.append(to_append)
    return after

def one_text_step(gen_text, ngrams, after, order):
    ngram = gen_text[-order:]
    if '@' in ngram:
        return gen_text
    ind = ngrams.index(ngram)
    return gen_text + np.random.choice(after[ind])

text = "Я скажу то, что для тебя не новость. Мир не такой уж солнечный и приветливый. Это очень опасное, жёсткое место. И если только дашь слабину, он опрокинет с такой силой тебя, что больше уже не встанешь. Ни ты, ни я, никто на свете не бьёт так сильно, как жизнь. Совсем не важно, как ты ударишь, а важно, какой держишь удар, как двигаешься вперёд. Будешь идти — иди, если с испугу не свернёшь. Только так побеждают! Если знаешь, чего ты стоишь, иди и бери своё, но будь готов удары держать, а не плакаться и говорить: «Я ничего не добился из-за него, из-за неё, из-за кого-то»! Так делают трусы, а ты не трус! Быть этого не может! …я всегда буду тебя любить, что бы ни случилось. Ты мой сын — плоть от плоти, самое дорогое, что у меня есть. Но пока ты не поверишь в себя, жизни не будет."


for order in range(2, 10):
    ngrams, count = make_ngrams(text, order)
    after = What_comes_after(text, ngrams, count, order)

    gen_text = text[:order]
    for x in range(2500):
        gen_text = one_text_step(gen_text, ngrams, after, order)
    print('\n', gen_text)