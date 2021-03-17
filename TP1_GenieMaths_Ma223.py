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
    n, m = A.shape(A)
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


print("\t\tQuestion 4\n\tEnsemble des graphiques fin du programme")
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


A = np.array([[2, 5, 6],[4, 11, 9],[-2 ,-8 , 7]])
B = np.array([[7], [12], [3]])

"""
def GaussChoixPivotTotal(A,B):
    n,m = np.shape(A)
    
    x = np.zeros(n)
    templ = [0,0]
    tempc = [0,0]
    jmax = 0
    kmax = 0
    for i in range(0, n-1):
        jmax = i
        kmax = i
        pivotmax= abs(A[jmax,kmax])

        if i >= 1:
            for k in range(i,n-1):
                for j in range(i,n-1):
                    if abs(A[j,k]) > pivotmax : 
                        jmax = j
                        kmax = k
                        pivotmax = abs(A[jmax,kmax])

            tempc[:] = A[:,i]
            A[:,i] = A[:,kmax]
            A[:,kmax] = tempc[:]
            templ[:] = A[i,:]
            A[i,:] = A[jmax,:]
            A[jmax,:] = templ[:]

        for k in range(i+1,n):
            g = A[k,i] / A[i,i]
            A[k, :] = A[k, :] - g * A[i, :]
    #Ax=B
    T = np.column_stack((A,B))
    x[n-1] = T[n-1,n] / T[n-1,n-1]

    for i in range(n-2, -1, -1):
        x[i] = T[i,n]
        for j in range(i+1, n):
            x[i]= x[i] - T[i,j] * x[j]
        x[i] =  x[i] / T[i,i]
    return x

 """

print("\t\t4.2 Pivot Total \n", "On a : ", GaussChoixPivotTotal(A,B))
print("---------------------------------------------")

#------------------------5. Graphique Comparatoire------------------------
"""
def Vect(n):
    B = np.random.rand(1, n)
    return B

def Mat(n):
    A = np.random.rand(n, n)
    return A
"""

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
    print("Solutions de Gauss :", nPV)
    stop_PV = time.time()
    tPV.append(stop_PV - start_PV)
    EPV.append(np.linalg.norm(np.dot(A,nPV)-np.ravel(B)))
    

    start_LU = time.time()
    nLU = LU(C, B)
    print("Solutions de LU :", nLU)
    stop_LU = time.time()
    tLU.append(stop_LU - start_LU)
    ELU.append(np.linalg.norm(np.dot(C,nLU)-np.ravel(B)))


    start_PVp = time.time()
    nPVp = GaussChoixPivotPartiel(D, B)
    print("Solutions de Pivot Partiel :", nPVp)
    stop_PVp = time.time()
    tPVp.append(stop_PVp - start_PVp)
    EPVp.append(np.linalg.norm(np.dot(D,nPVp)-np.ravel(B)))

"""
    start_PVt = time.time()
    nPVt = GaussChoixPivotPartiel(E, B)
    print("Solutions de Pivot Partiel :", nPVt)
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
#plt.plot(size_n, tPVt, color='cyan', label = 'PV')
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
#plt.plot(size_n, EPVt, color='cyan', label = 'PVp')
plt.xlabel('Taille de la matrice (n)')
plt.ylabel('||AX - B||')
plt.legend(loc='upper left')
plt.title("Estimation de l'erreur commise")
plt.show()


"""
B10 = Vect(10)
B50 = Vect(50)
B100 = Vect(100)
B150 = Vect(150)
B200 = Vect(200)
B250 = Vect(250)
B300 = Vect(300)
B350 = Vect(350)
B400 = Vect(400)
B450 = Vect(450)
B500 = Vect(500)

B = [B10, B50, B100, B150, B200, B250, B300, B350, B400, B450, B500]

A10 = Mat(10)
A50 = Mat(50)
A100 = Mat(100)
A150 = Mat(150)
A200 = Mat(200)
A250 = Mat(250)
A300 = Mat(300)
A350 = Mat(350)
A400 = Mat(400)
A450 = Mat(450)
A500 = Mat(500)

A = [A10, A50, A100, A150, A200, A250, A300, A350, A400, A450, A500]
C = np.copy(A)
D = np.copy(A)
E = np.copy(A)

#les listes temps



#matrice taille 10

start = time.time()
nPV10 = Gauss(A10, B10)
stop = time.time()
tPV.append(stop - start)

start = time.time()
nLU10 = LU(A10, B10)
stop = time.time()
tLU.append(stop - start)

start = time.time()
nPVp10 = GaussChoixPivotPartiel(A10, B10)
stop = time.time()
tPVp.append(stop - start)

#matrice taille 50

start = time.time()
nPV50 = Gauss(A50, B50)
stop = time.time()
tPV.append(stop - start)

start = time.time()
nLU50 = LU(A50, B50)
stop = time.time()
tLU.append(stop - start)

start = time.time()
nPVp50 = GaussChoixPivotPartiel(A50, B50)
stop = time.time()
tPVp.append(stop - start)

#matrice taille 100

start = time.time()
nPV100 = Gauss(A100, B100)
stop = time.time()
tPV.append(stop - start)

start = time.time()
nLU100 = LU(A100, B100)
stop = time.time()
tLU.append(stop - start)

start = time.time()
nPVp100 = GaussChoixPivotPartiel(A100, B100)
stop = time.time()
tPVp.append(stop - start)

#matrice taille 150

start = time.time()
nPV150 = Gauss(A150, B150)
stop = time.time()
tPV.append(stop - start)

start = time.time()
nLU150 = LU(A150, B150)
stop = time.time()
tLU.append(stop - start)

start = time.time()
nPVp150 = GaussChoixPivotPartiel(A150, B150)
stop = time.time()
tPVp.append(stop - start)

#matrice taille 200

start = time.time()
nPV200 = Gauss(A200, B200)
stop = time.time()
tPV.append(stop - start)

start = time.time()
nLU200 = LU(A200, B200)
stop = time.time()
tLU.append(stop - start)

start = time.time()
nPVp200 = GaussChoixPivotPartiel(A200, B200)
stop = time.time()
tPVp.append(stop - start)

#matrice taille 250

start = time.time()
nPV250 = Gauss(A250, B250)
stop = time.time()
tPV.append(stop - start)

start = time.time()
nLU250 = LU(A250, B250)
stop = time.time()
tLU.append(stop - start)

start = time.time()
nPVp250 = GaussChoixPivotPartiel(A250, B250)
stop = time.time()
tPVp.append(stop - start)

#matrice taille 300

start = time.time()
nPV300 = Gauss(A300, B300)
stop = time.time()
tPV.append(stop - start)

start = time.time()
nLU300 = LU(A300, B300)
stop = time.time()
tLU.append(stop - start)

start = time.time()
nPVp300 = GaussChoixPivotPartiel(A300, B300)
stop = time.time()
tPVp.append(stop - start)

#matrice taille 350

start = time.time()
nPV350 = Gauss(A350, B350)
stop = time.time()
tPV.append(stop - start)

start = time.time()
nLU350 = LU(A350, B350)
stop = time.time()
tLU.append(stop - start)

start = time.time()
nPVp350 = GaussChoixPivotPartiel(A350, B350)
stop = time.time()
tPVp.append(stop - start)


#matrice taille 400

start = time.time()
nPV400 = Gauss(A400, B400)
stop = time.time()
tPV.append(stop - start)

start = time.time()
nLU400 = LU(A400, B400)
stop = time.time()
tLU.append(stop - start)

start = time.time()
nPVp400 = GaussChoixPivotPartiel(A400, B400)
stop = time.time()
tPVp.append(stop - start)

#matrice taille 450

start = time.time()
nPV450 = Gauss(A450, B450)
stop = time.time()
tPV.append(stop - start)

start = time.time()
nLU450 = LU(A450, B450)
stop = time.time()
tLU.append(stop - start)

start = time.time()
nPVp450 = GaussChoixPivotPartiel(A450, B450)
stop = time.time()
tPVp.append(stop - start)

#matrice taille 500

start = time.time()
nPV500 = Gauss(A500, B500)
stop = time.time()
tPV.append(stop - start)

start = time.time()
nLU500 = LU(A500, B500)
stop = time.time()
tLU.append(stop - start)

start = time.time()
nPVp500 = GaussChoixPivotPartiel(A500, B500)
stop = time.time()
tPVp.append(stop - start)



#erreur PV

EPV10 = np.linalg.norm(A10.dot(nPV10)-B10)
EPV50 = np.linalg.norm(A50.dot(nPV50)-B50)
EPV100 = np.linalg.norm(A100.dot(nPV100)-B100)
EPV150 = np.linalg.norm(A150.dot(nPV150)-B150)
EPV200 = np.linalg.norm(A200.dot(nPV200)-B200)
EPV250 = np.linalg.norm(A250.dot(nPV250)-B250)
EPV300 = np.linalg.norm(A300.dot(nPV300)-B300)
EPV350 = np.linalg.norm(A350.dot(nPV350)-B350)
EPV400 = np.linalg.norm(A400.dot(nPV400)-B400)
EPV450 = np.linalg.norm(A450.dot(nPV450)-B450)
EPV500 = np.linalg.norm(A500.dot(nPV500)-B500)

EPV = [EPV10, EPV50, EPV100, EPV150, EPV200, EPV250, EPV300, EPV350, EPV400, EPV450, EPV500]

#erreur LU

ELU10 = np.linalg.norm(A10.dot(nLU10)-B10)
ELU50 = np.linalg.norm(A50.dot(nLU50)-B50)
ELU100 = np.linalg.norm(A100.dot(nLU100)-B100)
ELU150 = np.linalg.norm(A150.dot(nLU150)-B150)
ELU200 = np.linalg.norm(A200.dot(nLU200)-B200)
ELU250 = np.linalg.norm(A250.dot(nLU250)-B250)
ELU300 = np.linalg.norm(A300.dot(nLU300)-B300)
ELU350 = np.linalg.norm(A350.dot(nLU350)-B350)
ELU400 = np.linalg.norm(A400.dot(nLU400)-B400)
ELU450 = np.linalg.norm(A450.dot(nLU450)-B450)
ELU500 = np.linalg.norm(A500.dot(nLU500)-B500)

ELU = [ELU10, ELU50, ELU100, ELU150, ELU200, ELU250, ELU300, ELU350, ELU400, ELU450, ELU500]

#erreur PVp

EPVp10 = np.linalg.norm(A10.dot(nPVp10)-B10)
EPVp50 = np.linalg.norm(A50.dot(nPVp50)-B50)
EPVp100 = np.linalg.norm(A100.dot(nPVp100)-B100)
EPVp150 = np.linalg.norm(A150.dot(nPVp150)-B150)
EPVp200 = np.linalg.norm(A200.dot(nPVp200)-B200)
EPVp250 = np.linalg.norm(A250.dot(nPVp250)-B250)
EPVp300 = np.linalg.norm(A300.dot(nPVp300)-B300)
EPVp350 = np.linalg.norm(A350.dot(nPVp350)-B350)
EPVp400 = np.linalg.norm(A400.dot(nPVp400)-B400)
EPVp450 = np.linalg.norm(A450.dot(nPVp450)-B450)
EPVp500 = np.linalg.norm(A500.dot(nPVp500)-B500)

EPVp = [EPVp10, EPVp50, EPVp100, EPVp150, EPVp200, EPVp250, EPVp300, EPVp350, EPVp400, EPVp450, EPVp500]


#erreur PVt

EPVt10 = np.linalg.norm(A10.dot(nPVt10)-B10)
EPVt50 = np.linalg.norm(A50.dot(nPVt50)-B50)
EPVt100 = np.linalg.norm(A100.dot(nPVt100)-B100)
EPVt150 = np.linalg.norm(A150.dot(nPVt150)-B150)
EPVt200 = np.linalg.norm(A200.dot(nPVt200)-B200)
EPVt250 = np.linalg.norm(A250.dot(nPVt250)-B250)
EPVt300 = np.linalg.norm(A300.dot(nPVt300)-B300)
EPVt350 = np.linalg.norm(A350.dot(nPVt350)-B350)
EPVt400 = np.linalg.norm(A400.dot(nPVt400)-B400)
EPVt450 = np.linalg.norm(A450.dot(nPVt450)-B450)
EPVt500 = np.linalg.norm(A500.dot(nPVt500)-B500)

EPVt = [EPVt10, EPVt50, EPVt100, EPVt150, EPVt200, EPVt250, EPVt300, EPVt350, EPVt400, EPVt450, EPVt500]
"""
