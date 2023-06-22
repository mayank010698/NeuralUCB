import numpy as np
import pickle
import functools

def h1(x,a):
    return 10*(x@a)**2

def h2(x,A):
    return x.T@A.T@A@x

def h3(x,a):
    return np.cos(3*x@a)

#https://stackoverflow.com/questions/54544971/how-to-generate-uniform-random-points-inside-d-dimension-ball-sphere

def genXY(rnds=15000, ctx_size=4, dim=20):

    X = np.random.randn(rnds, ctx_size, dim)
    Xnorm = np.linalg.norm(X,axis=2)
    Xnorm = np.expand_dims(Xnorm, axis=2)
    Xsph = X/Xnorm
    random_radii = np.random.random((rnds, ctx_size, dim)) ** (1/dim)
    X = Xsph*random_radii

    a = np.random.randn(dim,1)
    a = a/np.linalg.norm(a)
    a = np.random.random((dim,1))**(1/dim)*a

    A = np.random.randn(dim,dim)

    y1 = h1(X,a)
    y1_noisy = y1 + np.random.randn(*y1.shape)

    h2_partial = lambda x:h2(x,A)
    y2 = np.apply_along_axis(h2_partial,arr=X,axis=2)
    y2_noisy = y2 + np.random.randn(*y2.shape)

    y3 = h3(X,a)
    y3_noisy = y3 + np.random.randn(*y3.shape)

    data = {}
    data["X"]  = X
    data["y1"] = y1_noisy
    data["y2"] = y2_noisy
    data["y3"] = y3_noisy

    data["a"] = a

    data["A"] = A

    with open("dataset.pickle","wb") as f:
        pickle.dump(data,f)
    
if __name__=="__main__":
    genXY(rnds = 15000, ctx_size = 4, dim = 20)
