'''
Calculate weights for compensation of unequal amount of pictures to use in Solver.py
Ignore differences from training data to validation data in AN Dataset, might be too complicated for what it is worth.
'''
import numpy as np

# AN_train, AN_val, CK and ISED weights multiplied with amount of CK/ISED pictures
Neutral=71687 + 477 + 0.05 * (1248 + 428)
Happiness=131465 + 489 + 0.09 * (1248 + 428)
Sadness=23863 + 482 + 0.35 * (1248 + 428)
Surprise=13510 + 474 + 0.15 * (1248 + 428)
Fear=5960 + 469 + 0.75 * (1248 + 428)
Disgust=3672 + 486 + 0.19 * (1248 + 428)
Anger=23377 + 475 + 0.45 * (1248 + 428)
Contempt=3668 + 495 + 1.3 * (1248 + 428)


Total=277202 + 3847 + 1248 + 428


def get_compensation_weights():
	return np.array([Total/Neutral, Total/Happiness, Total/Sadness, Total/Surprise, Total/Fear, Total/Disgust, Total/Anger, Total/Contempt])
