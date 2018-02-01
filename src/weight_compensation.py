'''
Calculate weights for compensation of unequal amount of pictures to use in Solver.py
Ignore differences from training data to validation data in AN Dataset, might be too complicated for what it is worth.
'''
import numpy as np

# AN_train, CK and ISED weights multiplied with amount of CK/ISED pictures
'''
Neutral=71687 + 477 + 0.05 * (1248 + 428)
Happiness=131465 + 489 + 0.09 * (1248 + 428)
Sadness=23863 + 482 + 0.35 * (1248 + 428)
Surprise=13510 + 474 + 0.15 * (1248 + 428)
Fear=5960 + 469 + 0.75 * (1248 + 428)
Disgust=3672 + 486 + 0.19 * (1248 + 428)
Anger=23377 + 475 + 0.45 * (1248 + 428)
Contempt=3668 + 495 + 1.3 * (1248 + 428)


Total=277202 + 3847 + 1248 + 428
'''


def get_AN_train_compensation_weights():
    Neutral = 71687
    Happiness = 131465
    Sadness = 23863
    Surprise = 13510
    Fear = 5960
    Disgust = 3672
    Anger = 23377
    Contempt = 3668

    #Total = 277202
    # return np.array([Total / Neutral, Total / Happiness, Total / Sadness, Total / Surprise, Total / Fear, Total / Disgust, Total / Anger, Total / Contempt])

    w = [Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger, Contempt]
    w = np.sum(w) / np.array(w)
    w = w / np.max(w)
    return w


def get_AN_val_compensation_weights():
    Neutral = 477
    Happiness = 489
    Sadness = 482
    Surprise = 474
    Fear = 469
    Disgust = 486
    Anger = 475
    Contempt = 495

    #Total = 3847
    # return np.array([Total / Neutral, Total / Happiness, Total / Sadness, Total / Surprise, Total / Fear, Total / Disgust, Total / Anger, Total / Contempt])

    w = [Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger, Contempt]
    w = np.sum(w) / np.array(w)
    w = w / np.max(w)
    return w
