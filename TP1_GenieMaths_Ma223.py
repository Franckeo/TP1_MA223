"""
TP1 Ma223_GénieMathématiques
Valentin Gazaix & Franck ELOY
"""
import numpy as np
import time
import matplotlib.pyplot as plt

#-----------------------------1. Introduction----------------------------
print("\t\t1. Introduction\n")
print("---------------------------------------------")
#------------------------2. L’algorithme de Gauss------------------------

#                           --- QUESTION 1 ---

A = np.array([[2, 5, 6],[4, 11, 9],[-2, -8, 7]])
B = np.array([[7], [12], [3]])

def ReductionGauss(A):
    n, m = np.shape(A)
    for i in range(0, n - 1):
        if A[i, i] == 0 :
            A[i,:] = A[i + 1]

        else :
            for j in range (i + 1, n):
                g = A[j, i] / A[i, i]
                A[j,:] = A[j,:] - g * A[i,:]
    return A

Taug = ReductionGauss(A)
print("---------------------------------------------")
print("\n\t\t2. L’algorithme de Gauss\n")
print("---------------------------------------------")
print("\t\tQuestion 1\n", Taug)
print("---------------------------------------------")

#                           --- QUESTION 2 ---

def ResolutionSystTriSup(Taug):
    n, m = Taug.shape
    x=[]
    #x = np.zeros(n)
    #x[n - 1] = Taug[n - 1, m - 1] / Taug[n - 1, n - 1]
    for i in range(n - 1):
        x.append(0)
    x.append(1)
    for i in range (n - 1, -1, -1):
        for j in range (n - 1, -1, -1):
            if i < j :
                A = Taug[i, j] * x[j]
                Taug[i, m - 1] = Taug[i, m - 1] - A
            if i == j :
                Taug[i][m - 1] = Taug[i][m - 1] / Taug[i, i]
        x[i] = Taug[i, m - 1]
    return(x)


print("\t\tQuestion 2\n", ResolutionSystTriSup(Taug))
print("---------------------------------------------")

#                           --- QUESTION 3 ---

def Gauss(A,B):
    Aaug = np.hstack([A, B])
    Ared = ReductionGauss(Aaug)

    return ResolutionSystTriSup(Ared)

A = np.array([[2, 5, 6],[4, 11, 9],[-2, -8, 7]])
B = np.array([[7], [12], [3]])

print("\t\tQuestion 3\nOn a :", Gauss(A, B))
print("---------------------------------------------")

#                           --- QUESTION 4 ---


print("\t\tQuestion 4\nEnsemble des graphiques a la fin du programme")
print("---------------------------------------------")

"""
def Vect(n):
    B = np.random.rand(n, 1)
    return B

def Mat(n):
    A = np.random.rand(n, n)
    return A
"""

#--------------------------3. Décomposition LU--------------------------

print("---------------------------------------------")
print("\n\t\t3. Decomposotion LU\n")
print("---------------------------------------------")

#                           --- QUESTION 1 ---

def DecompositionLU(A):
    n, n = np.shape(A)
    L = np.eye(n)
    U = np.copy(A)
    for i in range(0, n - 1):

        for j in range(i + 1, n):
            g = U[j, i] / U[i, i]
            L[j, i] = g
            U[j,:] = U[j,:] - g * U[i,:]

    return L, U

A = np.array([[2, 5, 6],[4, 11, 9],[-2, -8, 7]])
B = np.array([[7], [12], [3]])
(L, U) = DecompositionLU(A)

#                           --- QUESTION 2 ---


def ResolutionLU(L, U, B):
    n, n = np.shape(L)
    x = np.zeros(n)
    y = []

    for i in range(0, n):
        y.append(B[i])

        for k in range(0, i):
            y[i] = y[i] - L[i, k] * y[k]
        y[i] = y[i] / L[i, i]

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

def LU(A, B):
    n, n = np.shape(A)
    B = B.reshape(n, 1)
    L, U = DecompositionLU(A)
    return ResolutionLU(L, U, B)

print("\t\tQuestion 2\n","On a : ", ResolutionLU(L, U, B))
print("---------------------------------------------")

#-------------------4. Variantes de l’algorithme de Gaus-------------------
print("---------------------------------------------")
print("\n\t4. Variantes de l’algorithme de Gaus\n")
print("---------------------------------------------")
#                           --- QUESTION 1 ---

def GaussChoixPivotPartiel(A,B):
    A = np.column_stack ((A,B))
    n,m = A.shape
   
    for i in range (n):
       
        for j in range(0, n - i):
            l = abs(A[i + j, i])
            if l > abs(A[i, i]):
                T = A[i,:].copy()
                A[i,:] = A[i + j,:]
                A[i + j,:] = T
            p = A[i, i]

        if p != 0 :
            for k in range (i, n - 1):
                G = A[k + 1, i]/p
                A[k + 1,:] = A[k + 1,:] - G * A[i,:]

    Taug = A.copy()
    X = np.zeros(n)
   
    for i in range(n-1,-1,-1):
        S = 0
        for j in range (i + 1, n):
            S = S + Taug[i, j] * X[j]

        X[i] = (Taug[i, m - 1]-S) / Taug[i, i]

    return X


A = np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])
B = np.array([[7], [12], [3]])

print("\t\t4.1 Pivot Partiel \n", "On a : ", GaussChoixPivotPartiel(A, B))
print("---------------------------------------------")

#                           --- QUESTION 2 ---

"""
A = np.array([[2, 5, 6],[4, 11, 9],[-2 ,-8 , 7]])
B = np.array([[7], [12], [3]])




print("\t\t4.2 Pivot Total \n", "On a : ", GaussChoixPivotTotal(A,B))
print("---------------------------------------------")
"""
#------------------------5. Graphique Comparatoire------------------------

def Vect(n):
    B = np.random.rand(1, n)
    return B

def Mat(n):
    A = np.random.rand(n, n)
    return A


tPV = []
tLU = []
tPVp = []
tPVt = []

EPV =[]
ELU =[]
EPVp = [] 
EPVt = []

for n in range(1,502,50):
    A = np.random.rand(n, n)
    B = np.random.rand(n, 1)
    C = np.copy(A)
    D = np.copy(A)
    E = np.copy(A)

    start_PV = time.time()
    nPV = Gauss(A, B)
#    print("Solutions de Gauss :", nPV)
    stop_PV = time.time()
    tPV.append(stop_PV - start_PV)
    EPV.append(np.linalg.norm(np.dot(A,nPV)-np.ravel(B)))
    

    start_LU = time.time()
    nLU = LU(C, B)
#    print("Solutions de LU :", nLU)
    stop_LU = time.time()
    tLU.append(stop_LU - start_LU)
    ELU.append(np.linalg.norm(np.dot(C,nLU)-np.ravel(B)))


    start_PVp = time.time()
    nPVp = GaussChoixPivotPartiel(D, B)
#    print("Solutions de Pivot Partiel :", nPVp)
    stop_PVp = time.time()
    tPVp.append(stop_PVp - start_PVp)
    EPVp.append(np.linalg.norm(np.dot(D,nPVp)-np.ravel(B)))

"""
    start_PVt = time.time()
    nPVt = GaussChoixPivotPartiel(E, B)
#    print("Solutions de Pivot Total :", nPVt)
    stop_PVt = time.time()
    tPVt.append(stop_PVt - start_PVt)
    EPVt.append(np.linalg.norm(np.dot(E, nPVt)-np.ravel(B)))
"""
size_n = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

#                           --- GRAPH n/t ---

plt.style.use('dark_background')
plt.plot(size_n, tPV, color='red', label = 'PV')
plt.plot(size_n, tLU, color='lime', label = 'LU')
plt.plot(size_n, tPVp, color='blue', label = 'PVp')
#plt.plot(size_n, tPVt, color='cyan', label = 'PVt')
plt.xlabel('Taille de la matrice (n)')
plt.ylabel('Temps (s)')
plt.legend(loc='upper left')
plt.title('Temps de calcul CPU en fonction du nombre de matrice n')
plt.show()

#                           --- GRAPH n/E ---

plt.style.use('dark_background')
plt.plot(size_n, EPV, color='red', label = 'PV')
plt.plot(size_n, ELU, color='lime', label = 'LU')
plt.plot(size_n, EPVp, color='blue', label = 'PVp')
#plt.plot(size_n, EPVt, color='cyan', label = 'PVt')
plt.xlabel('Taille de la matrice (n)')
plt.ylabel('||AX - B||')
plt.legend(loc='upper left')
plt.title("Estimation de l'erreur commise")
plt.show()



