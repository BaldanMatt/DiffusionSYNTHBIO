import pathlib, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Latex Font
# Comment this if not using latex
plt.rcParams.update({
                    "text.usetex":True,
                    "font.family":"sans-serif",
                    "font.sans-serif":"Helvetica",
})

class Plotter():
    def __init__(self):
        self.figuresize = (2.33,2.33/1.33)

    def plot_tsne(
            self,
    ):
        pass

if  __name__ == "__main__":
    p = Plotter()
