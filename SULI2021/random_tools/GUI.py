import tkinter as tk
from import_mat import *

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        self.fake_stuff()


    def create_widgets(self):
        self.shot_select = tk.StringVar()
        self.shot_selector = tk.OptionMenu(self, *get_shot())

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")
        self.shot_selector.pack(side='top')

    def fake_stuff(self):
        print(self.shot_select.get())


root = tk.Tk()
app = Application(master=root)
app.mainloop()