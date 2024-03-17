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


def select_image(col=0, row=0, imgfr=None):
    global gray_img, fr1, tabs, l_imgFr, imgref
    if imgfr is None:
        imgfr = l_imgFr
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    colored_img = cv2.imread(file_path)
    if col == 1:
        imgref = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('ref.png', imgref)
        img1 = ctk.CTkImage(Image.open('ref.png'), size=image_size)
    
    else:
        gray_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('gray.png', gray_img)
        img1 = ctk.CTkImage(Image.open('gray.png'), size=image_size)
     
    # remove previous image
    for child in imgfr.winfo_children():
        info = child.grid_info()
        if info['row'] == row:
            child.destroy()

    ctk.CTkLabel(imgfr, image=img1, text='').grid(column=col, row=row, padx=10,pady=10)

    create_graph(tabs.tab('Shrink & Stretch'), 1, 0, img=gray_img)
    create_graph(tabs.tab('Shrink & Stretch'), 3, 0, txt='Processed Histogram')

    create_graph(tabs.tab(' Image Negative'), 1, 1, img=gray_img)
    create_graph(tabs.tab(' Image Negative'), 3, 1, txt='Negative Image Histogram')

    create_graph(tabs.tab('Linear Mapping'), 1, 0, img=gray_img)
    create_graph(tabs.tab('Linear Mapping'), 3, 0, txt='Linear Mapped Histogram')

    create_graph(tabs.tab('Non-Linear Mapping'), 1, 0, img=gray_img)
    create_graph(tabs.tab('Non-Linear Mapping'), 3, 0, txt='Non-Linear Mapped Histogram')

    create_graph(tabs.tab('ACE Filter'), 1, 0, img=gray_img)
    create_graph(tabs.tab('ACE Filter'), 3, 0, txt='ACE Histogram')

    if tabs.get() == 'Specified Equalization':
        for frm in tabs.tab('Specified Equalization').winfo_children():
            if frm.grid_info()['row'] == 1:
                fr = frm
        if col==0:
            create_graph(fr, 1, 0, img=gray_img, txt='Target Image Histogram')
        elif col==1:
            create_graph(fr, 3, 0, img=imgref, txt='Reference Image Histogram')

    global mapping
    mapping = np.arange(255)
    create_mapping_graph(fr1, 0, 2)


# ---------------Image Processing functions---------------
def negative_image():
    global gray_img, pro_img
    if gray_img is None:
        return
    neg_img = abs(np.amax(gray_img)-gray_img)
    cv2.imwrite('neg.png', neg_img)

    pro_img = ctk.CTkImage(Image.open('neg.png'), size=image_size)
    for child in r_imgFr.winfo_children():
        info = child.grid_info()
        if info['row'] == 0:
            child.destroy()
    ctk.CTkLabel(r_imgFr, image=pro_img, text='').grid(column=0, row=0, padx=10,pady=10)

    create_graph(tabs.tab(' Image Negative'), 3, 1, img=neg_img, txt='Negative Image Histogram')

def histogram_processing():
    global gray_img, Min, Max, pro_img
    
    if (not Min.get().isdigit() or not Max.get().isdigit()) or (int(Min.get()) < 0 or int(Max.get()) > 255) or (Min.get() == '' or Max.get() == '') or (int(Min.get()) >= int(Max.get())):
        messagebox.showerror('Invalid Range', 'Please enter valid range')
        return

    I_min = np.amin(gray_img)
    I_max = np.amax(gray_img)
    
    min = int(Min.get())
    max = int(Max.get())
    
    if min <= I_min and max >= I_max: # stretching
        
        stretched_img = np.round(((gray_img - I_min) / (I_max - I_min))* (max - min) + min)
        stretched_img = np.clip(stretched_img, 0, 255).astype('uint8')
        cv2.imwrite('stretched.png', stretched_img)

        pro_img = ctk.CTkImage(Image.open('stretched.png'), size=image_size)
        for child in r_imgFr.winfo_children():
            info = child.grid_info()
            if info['row'] == 0:
                child.destroy()
        ctk.CTkLabel(r_imgFr, image=pro_img, text='').grid(column=0, row=0, padx=10,pady=10)

        create_graph(tabs.tab('Shrink & Stretch'), 3, 0, img=stretched_img, txt='Stretched Histogram')

    elif min >= I_min and max <= I_max: # shrinking

        shrinked_img = np.round(((max - min) / (I_max - I_min))* (gray_img - I_min) + min)
        shrinked_img = np.clip(shrinked_img, 0, 255).astype('uint8')
        cv2.imwrite('shrinked.png', shrinked_img)
