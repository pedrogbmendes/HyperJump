from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


def expected_accuracy_loss(x1, std1, x2, std2):
    # c1 ~ N(x1, std1) config in SEL set
    # c2 ~ N(x2, std2) config in UNSEL set

    if std1 is not None and std2 is not None:
        # two untested configs
        # two gaussian distributions
        x = x1-x2
        std = std1+std2
        normal_dist = norm(x, std)

        eal = (x * (1 - normal_dist.cdf(0))) + ((std/np.sqrt(2*np.pi)) * np.exp(-(1/2)*((-x/std)**2)))

    elif std1 is None and std2 is not None:
        # x1 is tested
        # x2 is a normal distribution
        normal_dist = norm(x2, std2)

        eal = (x2 * (1 - normal_dist.cdf(x1))) + ((std2/np.sqrt(2*np.pi)) * np.exp(-(1/2)*(((x1-x2)/std2)**2))) - (x1 * (1 - normal_dist.cdf(x1)))

    elif std1 is not None and std2 is None:
        # x2 is tested
        # x1 is a normal distribution
        normal_dist = norm(x1, std1)

        eal = (x2 * normal_dist.cdf(x2)) - (x1 * normal_dist.cdf(x2)) + ((std1/np.sqrt(2*np.pi)) * np.exp(-(1/2)*(((x2-x1)/std1)**2)))

    else:
        # x2 is tested
        eal = 0

    print(" expected accuracy loss = " + str(eal))

    return eal


if __name__ == "__main__":

    x1 = 0.7
    std1 = 0.1

    x2 = 0.3
    std2 = None #0.1
    print(str(x1) + " " + str(std1) + "   -   ", str(x2) + " " + str(std2))
    
    ExpectedAccuracyLoss(x1, std1, x2, std2)

    x2 = 0.7
    std2 = None #0.7
    print(str(x1) + " " + str(std1) + "   -   ", str(x2) + " " + str(std2))
    
    ExpectedAccuracyLoss(x1, std1, x2, std2)


