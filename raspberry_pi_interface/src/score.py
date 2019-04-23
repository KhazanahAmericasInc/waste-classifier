#correct is True if the output was right and Flase if not
def append_score(correct):
    (num_correct,num_wrong,perc) = get_score() 
    if(correct):
	    num_correct+=1
    else:
	    num_wrong+=1
    f = open("score.txt","w")
    f.write(str(num_correct) + " " + str(num_wrong))
    f.close()

def get_score():
    f = open("score.txt", "r")
    score = f.read().split()
    num_correct = int(score[0])
    num_wrong = int(score[1])
    f.close()
    percent_corr = "null"
    if num_correct!=0 or num_wrong!=0:
	    percent_corr = int(num_correct/(num_correct+num_wrong)*100)
    return (num_correct,num_wrong, percent_corr)
