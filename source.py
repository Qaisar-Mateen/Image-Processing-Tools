import customtkinter as ctk
import cv2
import os
import threading
import numpy as np
from PIL import Image
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time

ctk.set_appearance_mode('dark')
ctk.set_default_color_theme('dark-blue')
