import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from time import sleep
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import threading
import queue

class Measurements():
    def __init__(self, filename_zx, filename_zy, title):
        self.filename_zx = filename_zx
        self.filename_zy = filename_zy
        self.title = title

    def run(self, queue):
        while True:
            grid_x, grid_y, grid_z = self.PrepareData(self.filename_zx, self.filename_zy)
            queue.put(grid_x)
            queue.put(grid_y)
            queue.put(grid_z)
            Zapelnienie = self.CheckV(grid_z)
            print(Zapelnienie)
            sleep(1)

    def ShowData(self):
        grid_x, grid_y, grid_z = self.PrepareData(self.filename_zx, self.filename_zy)
        V = self.CheckV(grid_z)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(grid_x, grid_y, -grid_z, cmap=cm.coolwarm, linewidth=0)
        ax.set_title(self.title)
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        plt.show()
    
    def PrepareData(self, filename_zx, filename_zy):
        data_zx, data_zy = self.ReadMeasurements(filename_zx, filename_zy)
        if len(data_zx) > 0 and len(data_zy) > 0:
            data_zx = self.StringToInteger(self.ConvertMeasurements(data_zx))
            data_zy = self.StringToInteger(self.ConvertMeasurements(data_zy))
            size_x, size_y = self.CheckSize(data_zx, data_zy)
            del data_zx[0]
            del data_zy[0]
            data_zx = self.CheckForFalseMeasurements(data_zx)
            data_zy = self.CheckForFalseMeasurements(data_zy)
            data_zx = list(reversed(data_zx)) 
            dist_x, dist_y = self.DistanceFromMiddle(data_zx, data_zy, size_x, size_y)
            data_zx, data_zy = self.ChangeHeight(data_zx, data_zy, size_x, size_y)

            dist_x = np.array(dist_x)
            dist_y = np.array(dist_y)
            data_zx = np.array(data_zx)
            data_zy = np.array(data_zy)

            grid_x, grid_y, zx, zy, X, Y = self.Create2Dgrid(data_zx, data_zy, dist_x, dist_y)
            grid_z = self.Create3Dgrid(zx, zy, X, Y)
        
            return grid_x, grid_y, grid_z
        else:
            return 0, 0, 0

    def CheckV(self, gridz):
        dlugoscY, dlugoscX = gridz.shape
        pom_z = 0
        suma_V = 0
        for i in range(1, dlugoscY):
            for j in range(1, dlugoscX):
                pom_z = (gridz[i][j] + gridz[i][j-1] + gridz[i-1][j] + gridz[i-1][j-1])/4
                suma_V += 100*100*pom_z
        suma_V = suma_V/1000000000
        Pole_calosci = dlugoscY*0.1*dlugoscX*0.1
        pom = 0
        for i in range(0, dlugoscY):
            for j in range(0, dlugoscX):
                if np.absolute(gridz[i][j]) > pom:
                   pom = np.absolute(gridz[i][j])
        pom = pom/1000
        V_calk = Pole_calosci * pom
        proc_zap = (suma_V/V_calk)*100

        return proc_zap
    
    def Create3Dgrid(self, zx, zy, X, Y):
        rows, cols = (len(Y), len(X))
        grid_z = [[0 for i in range(cols)] for j in range(rows)]
        grid_z = np.array(grid_z)
        middle_x = 0
        middle_y = 0
        if len(X) == len(zx) and len(Y) == len(zy):
            for i in range(0, len(X)):
                if X[i] == 0:
                    middle_x = i
            for i in range(0, len(Y)):
                if Y[i] == 0:
                    middle_y = i
            for i in range(0,len(Y)):
                grid_z[i][middle_y] = zy[i]
            for i in range(0, len(X)):
                grid_z[middle_x][i] = zx[i]
            for i in range(0, len(Y)):
                for j in range(0, len(X)):
                    grid_z[i][j] = (grid_z[middle_x][j] + grid_z[i][middle_y])/2
            grid_z = np.array(grid_z)
        return grid_z
    
    def Create2Dgrid(self, data_zx, data_zy, dist_x, dist_y):
        X = []
        Y = []
        plus_x = 0
        minus_x = 0
        plus_y = 0
        minus_y = 0
        
        for i in range(0, len(data_zx)):
            if dist_x[i] >= 0:
                if dist_x[i] > plus_x:
                    plus_x = dist_x[i]
            else:
                if dist_x[i] < minus_x:
                    minus_x = dist_x[i]
                    
        for i in range(0, len(data_zy)):
            if dist_y[i] >= 0:
                if dist_y[i] > plus_y:
                    plus_y = dist_y[i]
            else:
                if dist_y[i] < minus_y:
                    minus_y = dist_y[i]

        plus_x = int(self.Rounding(plus_x))
        minus_x = int(self.Rounding(minus_x))
        plus_y = int(self.Rounding(plus_y))
        minus_y = int(self.Rounding(minus_y))

        for i in range(minus_x, plus_x + 100, 100):
            X.append(i)
        for i in range(minus_y, plus_y + 100, 100):
            Y.append(i)
        
        X = np.array(X)
        Y = np.array(Y)
        grid_x, grid_y = np.meshgrid(X, Y)
        grid_x = np.array(grid_x)
        grid_y = np.array(grid_y)
        zx = []
        zy = []
        for i in range(0, len(X)):
            indeks_min = -1
            _min = 1000000000000000
            for j in range(0, len(dist_x)):
                if np.absolute(X[i] - dist_x[j]) < _min:
                    _min = np.absolute(X[i] - dist_x[j])
                    indeks_min = j
            zx.append(data_zx[indeks_min])

        for i in range(0, len(Y)):
            indeks_min = -1
            _min = 1000000000000000
            for j in range(0, len(dist_y)):
                if np.absolute(Y[i] - dist_y[j]) < _min:
                    _min = np.absolute(Y[i] - dist_y[j])
                    indeks_min = j
            zy.append(data_zy[indeks_min])

        return grid_x, grid_y, zx, zy, X, Y

    def Rounding(self, dist):
        pom = round(dist)
        if pom > 0:
            while pom % 100 != 0:
                pom-=1
        else:
            while pom % 100 != 0:
                pom+=1
        return pom
    
    def ChangeHeight(self, data_zx, data_zy, size_x, size_y):
        anglex = -((size_x - 1)/4)
        angley = -((size_y - 1)/4)

        for i in range(0, len(data_zx)):
            data_zx[i] = data_zx[i]*np.cos(np.deg2rad(anglex))
            anglex+=0.5
        for i in range(0, len(data_zy)):
            data_zy[i] = data_zy[i]*np.cos(np.deg2rad(angley))
            angley+=0.5

        return data_zx, data_zy
        
    def DistanceFromMiddle(self, data_zx, data_zy, size_x, size_y):
        anglex = -((size_x - 1)/4)
        angley = -((size_y - 1)/4)

        dist_x = np.zeros(len(data_zx))
        dist_y = np.zeros(len(data_zy))
        
        for i in range(0, len(data_zx)):
            dist_x[i] = data_zx[i]*np.sin(np.deg2rad(anglex))
            anglex+=0.5
            
        for i in range(0, len(data_zy)):
            dist_y[i] = data_zy[i]*np.sin(np.deg2rad(angley))
            angley+=0.5

        return dist_x, dist_y
    
    def CheckSize(self, data_zx, data_zy):
        pom1 = data_zx[0]
        pom2 = data_zy[0]
        return pom1, pom2
    
    def ReadMeasurements(self, filename_zx, filename_zy):
        reader = open(filename_zx, 'r')
        data_zx = reader.read()
        reader.close()
        reader = open(filename_zy, 'r')
        data_zy = reader.read()
        reader.close()

        return data_zx, data_zy

    def ConvertMeasurements(self, data):
        data = list(data.split(" "))
        return data
    
    def CheckForFalseMeasurements(self, data):
        horyzont = 15
        for i in range(1, len(data)-horyzont-1):
            suma = 0
            for j in range(i, i+horyzont):
                suma += data[j]
            srednia = suma/horyzont
            sumaKwadratow = 0
            for j in range(i, i+horyzont):
                sumaKwadratow += np.square(data[j] - srednia)
            odchStd = np.sqrt(sumaKwadratow/horyzont)
            for j in range(i, i+horyzont):
                if np.absolute(data[j]) > 3*odchStd:
                    data[j] = (data[j-1]+data[j+1])/2
        return data
        
    def StringToInteger(self, data):
        for i in range(0, len(data)):
            if data[i] == '':
                del data[i]
            else:
                data[i] = float(data[i])
        return data

def ReadQueue():
    if not queue1.empty():
        gridx = queue1.get()
        gridy = queue1.get()
        gridz = queue1.get()
        gridx = gridx/1000
        gridy = gridy/1000
        gridz = -gridz/1000
        
        ax.cla()
        ax.plot_surface(gridx, gridy, gridz, cmap=cm.coolwarm, linewidth=0)
        ax.set_title('Wykres testowy')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        graph.draw()
        
    main.after(200, ReadQueue)
        

if __name__ == '__main__':

    wykres = Measurements("192.168.0.922111.txt", "192.168.0.442112.txt", "Test")
    queue1 = queue.Queue()

    wykres_running = True

    thread1 = threading.Thread(target=wykres.run, args=(queue1,), daemon=True)
    thread1.start()
    
    #GUI#
    main = tk.Tk()
    main.title("Testowa wizualizacja")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    graph = FigureCanvasTkAgg(fig, master=main)
    canvas = graph.get_tk_widget()
    canvas.pack()

    ReadQueue()
    
    main.mainloop()

    main.quit()
    wykres_running = False
    thread1.join()







    



    
