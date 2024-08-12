import tkinter as tk

ventana = tk.Tk()
nombre_var = tk.StringVar()
entry_nombre = tk.Entry(ventana, textvariable=nombre_var)
entry_nombre.config(name='mi_entry')
entry_nombre.pack()
ventana.mainloop()