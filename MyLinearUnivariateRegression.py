class MyLinearUnivariateRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = 0.0


    def prodOf2(self,A,B):
        result = [[sum(a * b for a, b in zip(rowA,colB))
                   for colB in zip(*B)]
                   for rowA in A]
        return result

    def transposeMatrix(self,x):
        return [[x[row][col] for row in range(0, len(x))] for col in range(0, 3)]

    def getMatrixMinor(self,m, i, j):
        return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]

    def getMatrixDeterminant(self,m):
        if len(m) == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]

        determinant = 0
        for c in range(len(m)):
            determinant += ((-1) ** c) * m[0][c] * self.getMatrixDeterminant(self.getMatrixMinor(m, 0, c))
        return determinant

    def getMatrixInverse(self,m):
        determinant = self.getMatrixDeterminant(m)
        if len(m) == 2:
            return [[m[1][1] / determinant, -1 * m[0][1] / determinant],
                    [-1 * m[1][0] / determinant, m[0][0] / determinant]]

        adjunct = []
        for i in range(len(m)):
            adjunctRow = []
            for j in range(len(m)):
                minor = self.getMatrixMinor(m, i, j)
                adjunctRow.append(((-1) ** (i + j)) * self.getMatrixDeterminant(minor))
            adjunct.append(adjunctRow)
        inverse = self.transposeMatrix(adjunct)
        for i in range(len(inverse)):
            for j in range(len(inverse)):
                inverse[i][j] = inverse[i][j] / determinant
        return inverse

    def fit(self, x, y):
        xT=[[x[row][col] for row in range(0, len(x))] for col in range(0, 3)]
        prod=self.prodOf2(xT,x)
        Y=[]
        for i in range(len(y)):
            Y.append([y[i]])
        prodXY=self.prodOf2(xT,Y)
        xTx_1=self.getMatrixInverse(prod)
        B=self.prodOf2(xTx_1,prodXY)
        self.intercept_ = B[0]
        self.coef_ = B[1:]


    def predict(self, x):
        return [self.intercept_[0] + self.coef_[0][0] * val[0] + self.coef_[1][0] * val[1] for val in x]
