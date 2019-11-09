import matplotlib.pyplot as plt
import numpy as np
import os

def read_file(file_name, file_path="./results/"):
    rst = []
    with open(os.path.join(file_path, file_name), "r") as f:
        for l in f.readlines():
            number = l.split(" ")[-1]
            rst.append(float(number))
    return rst



def plot1():
    model1 = np.array(read_file("model1"))
    model2 = np.array(read_file("model2"))
    model3 = np.array(read_file("model3"))

    n = np.array(range(100))

    plt.plot(n, model1, "-", label='Teacher Forcing Rate = 0.5')
    plt.plot(n, model2, "-", label='Teacher Forcing Rate = 1.0')
    plt.plot(n, model3, "-", label='Teacher Forcing Rate = 0.0')

    plt.title('Loss Changes for Different Teacher Forcing Settings')
    plt.xlabel('number of epoches')
    plt.ylabel('loss per epoch')
    plt.legend(loc='upper right',
          fancybox=True, shadow=True)
    plt.show()

def plot2():
    model1 = np.array(read_file("model1"))
    model4 = np.array(read_file("model4"))
    model5 = np.array(read_file("model5"))
    model6 = np.array(read_file("model6"))
    n = np.array(range(100))
    plt.plot(n, model6, "-", label='No attention')
    plt.plot(n, model1, "-", label='attention = general')
    plt.plot(n, model4, "-", label='attention = dot')
    plt.plot(n, model5, "-", label='attention = concat')

    plt.title('Loss Changes for Different Attention Mechanism')
    plt.xlabel('number of epoches')
    plt.ylabel('loss per epoch')
    plt.legend(loc='upper right',
               fancybox=True, shadow=True)
    plt.show()


def plot3():
    model1 = np.array(read_file("model1"))/2.0
    model7 = np.array(read_file("model7"))/10.0
    n = np.array(range(100))
    plt.plot(n, model1, "-", label='batch size = 2')
    plt.plot(n, model7, "-", label='batch size = 10')

    plt.title('Loss Changes for Different Batch Sizes')
    plt.xlabel('number of epoches')
    plt.ylabel('loss per epoch')
    plt.legend(loc='upper right',
               fancybox=True, shadow=True)
    plt.show()


if __name__ == "__main__":
    plot3()
