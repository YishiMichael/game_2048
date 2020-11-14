import tkinter as tk
import tkinter.font as tkFont
import numpy as np
import copy
root = tk.Tk()

class Main:
    def __init__(self, parent):
        self.parent = parent
        self.viewmap = np.zeros(16, dtype='int64')
        self.buttonmap = []
        self.style = None
        self.start_game()

    def order(self, x, y):
        n = x * 4
        n += y
        return n
        
    def re_order(self, n):
        x = n // 4
        y = n % 4
        return np.array([x, y], dtype='int8')

    def start_game(self):
        self.parent.title('Game of 2048')
        mine_location = np.random.choice(np.arange(16), 2, replace=False)
        for i in mine_location:
            self.viewmap[i] = 1
        for i in range(4):
            for j in range(4):
                n = self.order(i, j)
                
                display_font = tkFont.Font(family='Helvetica', size=18, weight=tkFont.BOLD)
                lbl = tk.Label(self.parent, width=6, height=3, text=str(2**self.viewmap[n]), relief='groove', font=display_font)
                if self.viewmap[n] == 0:
                    lbl['text'] = ' '
                
                lbl.grid(row=i, column=j)
                self.buttonmap.append(lbl)

        self.parent.bind('<Up>', self.north)
        self.parent.bind('<Down>', self.south)
        self.parent.bind('<Left>', self.west)
        self.parent.bind('<Right>', self.east)

        self.parent.mainloop()

    def update_map(self):
        for n in range(16):
            if self.viewmap[n] == 0:
                self.buttonmap[n].configure(text=' ')
            else:
                self.buttonmap[n].configure(text=str(2**self.viewmap[n]))
        
    def action(self):
        old_map = copy.copy(self.re)
        self.re = self.re.T
        for i in range(4):
            if np.sum(self.re[i]) != 0:
                s = np.array([], dtype='int64')
                for j in self.re[i]:
                    if j != 0:
                        s = np.append(s, j)
                if np.size(s) != 1:
                    if np.size(s) ==4 and s[0] == s[1] and s[2] == s[3]:
                        s[0] += 1
                        s[2] += 1
                        s = np.delete(s, 1)
                        s = np.delete(s, 2)
                    elif s[0] == s[1]:
                        s[0] += 1
                        s = np.delete(s, 1)
                    elif np.size(s) >= 3 and s[1] == s[2]:
                        s[1] += 1
                        s = np.delete(s, 2)
                    elif np.size(s) == 4 and s[2] == s[3]:
                        s[2] += 1
                        s = np.delete(s, 3)
                while np.size(s) != 4:
                    s = np.append(s, 0)
                self.re[i] = s
        self.re = self.re.T
        new_map = copy.copy(self.re)
        judge = 0
        for i in range(4):
            for j in range(4):
                if old_map[i][j] == new_map[i][j]:
                    judge += 1
        if judge == 16:
            return False
        else:
            return True

    def new_num(self):
        choices = np.array([], dtype='int8')
        for i in range(16):
            if self.viewmap[i] == 0:
                choices = np.append(choices, i)
        if np.size(choices) == 0:
            pass
        else:
            new_digit = np.random.choice(choices, 1)
            self.viewmap[new_digit] = 1
        self.viewmap = copy.deepcopy(self.viewmap.reshape(16))
        self.update_map()

    def east(self, a):
        self.re = self.viewmap.reshape((4, 4))
        self.re = self.re.T[::][::-1]
        if self.action():
            self.re = self.re[::][::-1].T
            self.new_num()
        else:
            self.re = self.re[::][::-1].T

    def south(self, a):
        self.re = self.viewmap.reshape((4, 4))
        self.re = self.re[::][::-1]
        if self.action():
            self.re = self.re[::][::-1]
            self.new_num()
        else:
            self.re = self.re[::][::-1]

    def west(self, a):
        self.re = self.viewmap.reshape((4, 4))
        self.re = self.re.T
        if self.action():
            self.re = self.re.T
            self.new_num()
        else:
            self.re = self.re.T
        
    def north(self, a):
        self.re = self.viewmap.reshape((4, 4))
        if self.action():
            self.new_num()
        

Main(root)
