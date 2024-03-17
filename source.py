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

# Global variables
k1_val,k2_val,process_btn,upload_btn,winSize,imgref,wtf = 0.5,0.5,None,None,None,None,False
upperFr,mapping,fr1,k,r,arrow,app,gray_img,r_imgFr,l_imgFr,tabs,Min,Max = None,None,None,None,None,None,None,None,None,None,None,None,None
image_size = (339, 190)
hov, nor = '#AF4BD6', '#9130BD'


def event_handler():
    global app, tabs, gray_img, l_imgFr, upperFr, wtf, pro_img, r_imgFr
    event = False
    tab = tabs.get()
    change = False
    while app:
        if tab != tabs.get() and wtf == False:
            if tab == 'Specified Equalization':
                change = True
            tab = tabs.get()
            event = True

        if wtf:
            wtf = False
            if os.path.exists('equalized.png'):
                    pro_img = ctk.CTkImage(Image.open('equalized.png'), size=image_size)
                    ctk.CTkLabel(r_imgFr, image=pro_img, text='').grid(column=0, row=0, padx=10, pady=10)
            
        if tabs.get() == 'Specified Equalization':
            if event:
                event = False

                for arrow in upperFr.winfo_children():
                    if arrow.grid_info()['column'] == 3 and arrow.grid_info()['row'] == 0:
                        arrow.destroy()
            
                upload_btn.destroy()
                
                for child in l_imgFr.winfo_children():
                    child.destroy()

                l_imgFr.destroy()
                r_imgFr.destroy() 
                r_imgFr = ctk.CTkFrame(upperFr, corner_radius=20, fg_color="#2B2B2B")
                r_imgFr.grid(row=0, column=5, padx=15, pady=15, sticky='snew')
    
                if pro_img is not None:
                    ctk.CTkLabel(r_imgFr, image=pro_img, text='').grid(column=0, row=0, padx=10, pady=10)
                else:
                    img1 = ctk.CTkImage(Image.open('no image.png'), size=image_size)
                    ctk.CTkLabel(r_imgFr, image=img1, text='').grid(column=0, row=0, padx=10, pady=10)

                ctk.CTkLabel(r_imgFr, text='Processed Image').grid(column=0, row=1, padx=10, pady=(0,10))
                
        else:
            if event:
                event = False
                if change:
                    uploaded = True if gray_img is not None else False 
                    upper_frame(uploaded)
                    if uploaded:
                        img1 = ctk.CTkImage(Image.open('gray.png'), size=image_size)
                        ctk.CTkLabel(l_imgFr, image=img1, text='').grid(column=0, row=0, padx=10,pady=10)
                        
                    change = False

        time.sleep(0.15)

