import numpy as np
import matplotlib.pyplot as plt


def single_people_number(user, n_groups):
    neutral_number = user[0]
    std_neutral = ()

    happy_number = user[1]
    std_happy = ()

    sad_number = user[2]
    std_sad = ()

    angry_number = user[3]
    std_angry = ()

    fear_number = user[4]
    std_fear = ()

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.15

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, neutral_number, bar_width,
                    alpha=opacity, color='#999999',
                    # yerr=std_men,
                    error_kw=error_config,
                    label='Neutral')

    rects2 = ax.bar(index + bar_width, happy_number, bar_width,
                    alpha=opacity, color='#FFFF00',
                    # yerr=std_men,
                    error_kw=error_config,
                    label='Happy')

    rects3 = ax.bar(index + 2 * bar_width, sad_number, bar_width,
                    alpha=opacity, color='blue',
                    # yerr=std_women,
                    error_kw=error_config,
                    label='Sad')

    rects4 = ax.bar(index + 3 * bar_width, angry_number, bar_width,
                    alpha=opacity, color='red',
                    # yerr=std_men,
                    error_kw=error_config,
                    label='Angry')

    rects5 = ax.bar(index + 4 * bar_width, fear_number, bar_width,
                    alpha=opacity, color='green',
                    # yerr=std_women,
                    error_kw=error_config,
                    label='Fear')

    ax.set_xlabel('The process of imitating four expressions')
    ax.set_ylabel('Number of expressions')
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(index + 2 * bar_width)
    ax.set_xticklabels(('Happy Imitate', 'Sad Imitate', 'Angry Imitate', 'Fear Imitate'))
    ax.legend()

    fig.tight_layout()
    # plt.savefig('E:\\cloud\\cuhksz\\mini_Xception\\level5\\tfApp\image\\test.png')
    plt.show()
    return 0


def get_single_emotion(Emotion_file):

    happy = []
    sad = []
    angry = []
    fear = []
    neutral = []
    emotions_result = Emotion_file

    for i in range(len(emotions_result.Emotion)):
        if emotions_result.Emotion[i] == 'happy':
            # print emotions_result.Emotion[i]
            happy.append(emotions_result.Emotion[i])
        elif emotions_result.Emotion[i] == 'sad':
            # print emotions_result.Emotion[i]
            sad.append(emotions_result.Emotion[i])
        elif emotions_result.Emotion[i] == 'angry':
            # print emotions_result.Emotion[i]
            angry.append(emotions_result.Emotion[i])
        elif emotions_result.Emotion[i] == 'fear':
            # print emotions_result.Emotion[i]
            fear.append(emotions_result.Emotion[i])
        else:
            # print emotions_result.Emotion[i]
            neutral.append(emotions_result.Emotion[i])

    return happy,sad,angry,fear,neutral


def get_single_probability(Emotion_file):

    happy_p = []
    sad_p = []
    angry_p = []
    fear_p = []
    neutral_p = []
    emotions_result = Emotion_file

    for i in range(len(emotions_result.Emotion)):
        if emotions_result.Emotion[i] == 'happy':
            # print emotions_result.Probability[i]
            happy_p.append(emotions_result.Probability[i])
        elif emotions_result.Emotion[i] == 'sad':
            # print emotions_result.Probability[i]
            sad_p.append(emotions_result.Probability[i])
        elif emotions_result.Emotion[i] == 'angry':
            # print emotions_result.Probability[i]
            angry_p.append(emotions_result.Probability[i])
        elif emotions_result.Emotion[i] == 'fear':
            # print emotions_result.Probability[i]
            fear_p.append(emotions_result.Probability[i])
        else:
            # print emotions_result.Probability[i]
            neutral_p.append(emotions_result.Probability[i])

    return happy_p,sad_p,angry_p,fear_p,neutral_p


# getting 4 kind of emotion values
def sigle_emotion_label(emotions_result, label):
# emotions_result is file
# return each emotion label 
    Emotion = label
    emotion = []
    for i in range(len(emotions_result.id)):
        if emotions_result.Emotion[i] == Emotion:
            emotion.append(emotions_result.Emotion[i])
        else:
            emotion.append(0)
    # print len(emotion), emotion
    return emotion

def sigle_emotion_probability(emotions_result, label):
# emotions_result is file
# return each emotion label 
    Emotion = label
    emotion_p = []
    for i in range(len(emotions_result.id)):
        if emotions_result.Emotion[i] == Emotion:
            emotion_p.append(emotions_result.Probability[i])
        else:
            emotion_p.append(0)
    # print emotion_p
    return emotion_p