import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

dummy_data = [(-1,0), (0,1), (1,0)]
X, y = zip(*dummy_data)
X = np.array(X).reshape(-1,1)
y = np.array(y)

m=len(X)
gamma = 0

def gaussian_kernel(distances):
    weights = np.exp(-gamma*(distances**2))
    return weights/np.sum(weights)

plt.rc('font', size = 18)
plt.rcParams['figure.constrained_layout.use'] = True

for g in [0,1,5,10,25]:
    gamma = g
    model = KNeighborsRegressor(n_neighbors=m,weights=gaussian_kernel).fit(X, y)
    Xtest = np.linspace(-3, 3, num=1000).reshape(-1, 1)
    ypred = model.predict(Xtest)
    plt.plot(Xtest, ypred, label=f"Gamma: {g}")

plt.scatter(X, y, color='red', marker='+', label="Training Data")
plt.xlabel("input x"); plt.ylabel("output y")
plt.legend()
plt.title("K Neighbours Regression")
plt.show()

#############################(c)###############################

for C in [0.1, 1, 1000]:
    plt.rc('font', size = 18)
    plt.rcParams['figure.constrained_layout.use'] = True

    for g in [0,1,5,10,25]:
        model = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=g).fit(X, y)
        Xtest=np.linspace(-3,3,num=1000).reshape(-1, 1)
        ypred = model.predict(Xtest)
        print("g:", g, "dual coef:", model.dual_coef_)
        plt.plot(Xtest, ypred, label=f"Gamma: {g}")

    plt.scatter(X, y, color='red', marker='+', label="Training Data")
    plt.xlabel("input x")
    plt.ylabel("output y")
    plt.legend()
    plt.title(f"Kernel Ridge C: {C}")
    plt.show()

#################################(ii)(a)###############################

df = pd.read_csv("week6.csv", comment='#')
X = np.array(df.iloc[:,0]).reshape(-1,1)
y = np.array(df.iloc[:,1])
m=len(X)

plt.rc('font', size = 18)
plt.rcParams['figure.constrained_layout.use'] = True

for g in [0,1,5,10,25]:
    gamma = g
    model = KNeighborsRegressor(n_neighbors=m,weights=gaussian_kernel).fit(X, y)
    Xtest = np.linspace(-3, 3, num=1000).reshape(-1, 1)
    ypred = model.predict(Xtest)
    plt.plot(Xtest, ypred, label=f"Gamma: {g}")

plt.scatter(X, y, color='red', marker='+', label="Training Data")
plt.xlabel("input x"); plt.ylabel("output y")
plt.legend()
plt.title("K Neighbours Regression")
plt.show()

#############################(ii)(b)###############################

C = 1

for g in [0,1,5,10,25]:
    model = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=g).fit(X, y)
    Xtest=np.linspace(-3,3,num=1000).reshape(-1, 1)
    ypred = model.predict(Xtest)
    plt.plot(Xtest, ypred, label=f"Gamma: {g}")

plt.scatter(X, y, color='red', marker='+', label="Training Data")
plt.xlabel("input x")
plt.ylabel("output y")
plt.legend()
plt.title(f"Kernel Ridge C: {C}")
plt.show()


#################################(ii)(c)################################
#KNN Regression with Gaussian Kernel Weights

m_split = int(len(X)*(4/5))
mean_err = []; std_err = []
gamma_range = [0,1,5,10,25]


for g in gamma_range:
    gamma = g
    temp = []
    #n_neighbours can only be as big as n_samples in cross val split
    model = KNeighborsRegressor(n_neighbors=m_split, weights=gaussian_kernel).fit(X, y)

    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        temp.append(mean_squared_error(y[test], ypred))
    mean_err.append(np.array(temp).mean())
    std_err.append(np.array(temp).std())

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
plt.errorbar(gamma_range, mean_err, yerr=std_err)
plt.xlabel('gamma'); plt.ylabel('Mean square error')
plt.xlim((-0.5, 25.5))
plt.title('Cross Validation: kNN')
plt.show()

#--------------Kernelised Ridge Regression (gamma)------------------------

mean_err = []; std_err = []
gamma_range = [0.1,0.5,1,5,10,25]
C = 1

for g in gamma_range:
    model = KernelRidge(alpha=1.0/C, kernel='rbf', gamma=g).fit(X, y)
    temp = []

    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        #print("intercept ", m.intercept_, "slope ", m.coef_,
        # " square error ", mean_squared_error(y[test], ypred))
        temp.append(mean_squared_error(y[test], ypred))
    mean_err.append(np.array(temp).mean())
    std_err.append(np.array(temp).std())

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
plt.errorbar(gamma_range, mean_err, yerr=std_err)
plt.xlabel('gamma'); plt.ylabel('Mean square error')
plt.xlim((-0.5, 25.5))
plt.title('Cross Validation: Kernel ridge')
plt.show()

#---------------------Kernel Ridge (alpha)---------------------------------

mean_err = []; std_err = []
g = 25
C_range = [0.01,0.1,1,10,1000]
#C_range = [0.01,0.05,0.1,0.5,1,5]

for C in C_range:
    model = KernelRidge(alpha=1.0/C, kernel='rbf', gamma=g).fit(X, y)
    temp = []

    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        #print("intercept ", m.intercept_, "slope ", m.coef_,
        # " square error ", mean_squared_error(y[test], ypred))
        temp.append(mean_squared_error(y[test], ypred))
    mean_err.append(np.array(temp).mean())
    std_err.append(np.array(temp).std())

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
plt.errorbar(C_range, mean_err, yerr=std_err)
plt.xlabel('C'); plt.ylabel('Mean square error')
plt.xlim((-5, 1005))
#plt.xlim((-.05, 5.05))
plt.title('Cross Validation: Kernel ridge')
plt.show()

#---------------------------------------------------

for C in [0.01, 1, 1000]:
    plt.rc('font', size = 18)
    plt.rcParams['figure.constrained_layout.use'] = True
    g = 25

    model = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=g).fit(X, y)
    Xtest=np.linspace(-3,3,num=1000).reshape(-1, 1)
    ypred = model.predict(Xtest)
    print("g:", g, "dual coef:", model.dual_coef_)
    plt.plot(Xtest, ypred, label=f"Gamma: {g}")

    plt.scatter(X, y, color='red', marker='+', label="Training Data")
    plt.xlabel("input x")
    plt.ylabel("output y")
    plt.legend()
    plt.title(f"Kernel Ridge C: {C}")
    plt.show()

#------------------predictions--------------------------------

gamma = 25
C = 1
predict_values = np.linspace(-2,2,41).reshape(-1,1)

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
plt.scatter(X, y, color='red', marker='+', label="Dummy")

model = KNeighborsRegressor(n_neighbors=m,weights=gaussian_kernel).fit(X, y)
ypred = model.predict(predict_values)
plt.plot(predict_values, ypred, label="kNN", color="navy", marker=".")

model2 = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=gamma).fit(X, y)
ypred2 = model2.predict(predict_values)
plt.plot(predict_values, ypred2, label="Kernel Ridge", color="orange", marker=".")

plt.xlabel("input x"); plt.ylabel("output y")
plt.legend()
plt.title("KNN vs Kernel Ridge")
plt.show()

