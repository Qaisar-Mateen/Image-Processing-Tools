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
pro_img = None


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

    create_graph(tabs.tab('Blur Filter'), 1, 0, img=gray_img)
    create_graph(tabs.tab('Blur Filter'), 3, 0, txt='Blur Image Histogram')
    
    create_graph(tabs.tab('Denoise Filter'), 1, 0, img=gray_img)
    create_graph(tabs.tab('Denoise Filter'), 3, 0, txt='Denoised Image Histogram')


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

        pro_img = ctk.CTkImage(Image.open('shrinked.png'), size=image_size)
        for child in r_imgFr.winfo_children():
            info = child.grid_info()
            if info['row'] == 0:
                child.destroy()
        ctk.CTkLabel(r_imgFr, image=pro_img, text='').grid(column=0, row=0, padx=10,pady=10)

        create_graph(tabs.tab('Shrink & Stretch'), 3, 0, img=shrinked_img, txt='Shrinked Histogram')
    
    else:
        messagebox.showerror('Invalid Range', 'Range is neither shrinking nor stretching')
        return


def equation_mapping():
    global gray_img, mapping, pro_img

    mapped_img = mapping[gray_img].astype('uint8')
    cv2.imwrite('mapped.png', mapped_img)

    pro_img = ctk.CTkImage(Image.open('mapped.png'), size=image_size)
    for child in r_imgFr.winfo_children():
        info = child.grid_info()
        if info['row'] == 0:
            child.destroy()
    ctk.CTkLabel(r_imgFr, image=pro_img, text='').grid(column=0, row=0, padx=10,pady=10)

    create_graph(tabs.tab('Linear Mapping'), 3, 0, img=mapped_img, txt='Mapped Image Histogram')


def Non_Linear_Mapping():
    global k, r, pro_img
    k_val = k.get()
    r_val = r.get()

    if (k_val == '' or r_val == ''):
        messagebox.showerror('Invalid Input', 'Please enter valid input')
        return
        
    k_val = float(k_val)
    r_val = float(r_val)

    if (not isinstance(k_val, int) and not isinstance(k_val, float)) or (not isinstance(k_val, int) and not isinstance(k_val, float)) or (float(k_val) <= 0 or float(r_val) == 0):
        messagebox.showerror('Invalid Input', 'Please enter valid input')
        return
    
    if r_val < 0:
        r_val = abs(r_val)
        r_val = 1 / r_val

    mapped_img = np.round(k_val * ((gray_img/k_val) ** r_val))
    mapped_img = np.clip(mapped_img, 0, 255).astype('uint8')
    cv2.imwrite('non_linear_mapped.png', mapped_img)

    pro_img = ctk.CTkImage(Image.open('non_linear_mapped.png'), size=image_size)
    
    for child in r_imgFr.winfo_children():
        info = child.grid_info()
        if info['row'] == 0:
            child.destroy()

    ctk.CTkLabel(r_imgFr, image=pro_img, text='').grid(column=0, row=0, padx=10,pady=10)

    create_graph(tabs.tab('Non-Linear Mapping'), 3, 0, img=mapped_img, txt='Mapped Image Histogram')


def ace_filter():
    global pro_img
    
    if (not winSize.get().isdigit()) or (winSize.get() == '') or (int(winSize.get()) < 1):
        messagebox.showerror('Invalid Input', 'Please enter valid input')
        return
    
    global gray_img, r_imgFr, k1_val, k2_val
    
    size = int(winSize.get())
    
    img = np.float32(gray_img)

    m = cv2.blur(img, (size, size)) # calculate local mean for each window
    std = cv2.blur((img - m)**2, (size, size))**0.5 # calculate local std for each window
    mean = np.mean(img)

    # k1(M(r,c)/local std)(I(r,c) - m(r,c)) + k2m(r,c)
    ace_image = k1_val *(mean/std)*(img - m) + k2_val * m
    ace_image = np.clip(ace_image, 0, 255).astype('uint8')
    cv2.imwrite('ace.png', ace_image)
    
    pro_img = ctk.CTkImage(Image.open('ace.png'), size=image_size)
    for child in r_imgFr.winfo_children():
        info = child.grid_info()
        if info['row'] == 0:
            child.destroy()

    ctk.CTkLabel(r_imgFr, image=pro_img, text='').grid(column=0, row=0, padx=10,pady=10)
    create_graph(tabs.tab('ACE Filter'), 3, 0, img=ace_image, txt='ACE Histogram')


def specified_equalization():
    global gray_img, imgref, tabs, r_imgFr, wtf
    if imgref is None:
        messagebox.showerror('ERROR', 'Please select both images first!')
        return
    
    # running sum of the histogram
    values_target = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    values_target = values_target.cumsum()
    
    values_ref = cv2.calcHist([imgref], [0], None, [256], [0, 256])
    values_ref = values_ref.cumsum()

    # normalizing the values
    values_target = values_target / values_target.max()
    values_ref = values_ref / values_ref.max()

    # multiply by max gray level
    values_target = np.round(values_target * np.amax(gray_img)).astype('uint8')
    values_ref = np.round(values_ref * np.amax(imgref)).astype('uint8')

    # mapping the values
    new_img = np.interp(values_target, values_ref, range(256)).astype('uint8')

    new_img = new_img[gray_img]

    cv2.imwrite('equalized.png', new_img)

    wtf = True

    for child in r_imgFr.winfo_children():
        info = child.grid_info()
        if info['row'] == 0 and info['column'] == 0:
            child.destroy()
    
    for frm in tabs.tab('Specified Equalization').winfo_children():
            if frm.grid_info()['row'] == 1:
                fr = frm

    create_graph(fr, 5, 0, img=new_img, txt='Processed Histogram')


def Blur_filter():
    global gray_img, winSize1, pro_img
    if (not winSize1.get().isdigit()) or (winSize1.get() == '') or (int(winSize1.get()) < 1):
        messagebox.showerror('Invalid Input', 'Please enter valid input')
        return
    
    size = int(winSize1.get())

    # padding the image
    padded_img = np.pad(gray_img, ((size//2, size//2), (size//2, size//2)), mode='edge')

    new_img = np.zeros(gray_img.shape, dtype=np.uint8)

    def pseudo_median_filter(s):
        
        def maximin(s, m):
            return max([min(s[i:m+i]) for i in range(len(s) - m + 1)])
        
        def minimax(s, m):
            return min([max(s[i:m+i]) for i in range(len(s) - m + 1)])
        
        win = np.copy(s)
        s = s.flatten()
        m = np.round((len(s)+1)/2).astype(int)

        return int(minimax(s, m)/2 + maximin(s, m)/2)

    
    height, width = gray_img.shape
    size = size // 2
    new_img = np.copy(gray_img)
    
    for y in range(size, height - size):
        for x in range(size, width - size):
            
            window = padded_img[y - size:y + size + 1, x - size:x + size + 1]
            window = pseudo_median_filter(window)

            new_img[y, x] = window
    
    new_img = new_img[size//2:-size//2, size//2:-size//2]
    cv2.imwrite('blurred.png', new_img)

    pro_img = ctk.CTkImage(Image.open('blurred.png'), size=image_size)
    for child in r_imgFr.winfo_children():
        info = child.grid_info()
        if info['row'] == 0:
            child.destroy()
    ctk.CTkLabel(r_imgFr, image=pro_img, text='').grid(column=0, row=0, padx=10,pady=10)
    create_graph(tabs.tab('Blur Filter'), 3, 0, img=new_img, txt='Blurred Image Histogram')


def Denose_filter():
    global gray_img, winSize2, variance, pro_img
    try:
        size = int(winSize2.get())
        noise_variance = float(variance.get())
        if size < 1 or noise_variance < 0:
            raise ValueError
    except ValueError:
        messagebox.showerror('Invalid Input', 'Please enter valid input')
        return

    # padding the image
    padded_img = np.pad(gray_img, ((size//2, size//2), (size//2, size//2)), mode='edge')

    new_img = np.copy(padded_img)

    h, w = gray_img.shape

    for y in range(size, h - size):
        for x in range(size, w - size):
            window = padded_img[y - size:y + size + 1, x - size:x + size + 1]
            mean = np.mean(window.flatten())
            var = np.var(window.flatten())
            #print(new_img[y, x], window[size, size])
            print(noise_variance/var)
            new_img[y, x] = (new_img[y, x] - (noise_variance/var)*(new_img[y, x] - mean)).astype('uint8')

    # to remove added padding
    new_img = new_img[size//2:-size//2, size//2:-size//2]
    cv2.imwrite('denois.png', new_img)

    pro_img = ctk.CTkImage(Image.open('denois.png'), size=image_size)
    for child in r_imgFr.winfo_children():
        info = child.grid_info()
        if info['row'] == 0:
            child.destroy()
    ctk.CTkLabel(r_imgFr, image=pro_img, text='').grid(column=0, row=0, padx=10,pady=10)
    create_graph(tabs.tab('Denoise Filter'), 3, 0, img=new_img, txt='Denoised Image Histogram')



def process_image():
    global tabs, gray_img
    if gray_img is None:
        messagebox.showerror('ERROR', 'Please select an image first!')
        return
     
    if tabs.get() == ' Image Negative':
        negative_image()
    elif tabs.get() == 'Shrink & Stretch':
        histogram_processing()
    elif tabs.get() == 'Linear Mapping':
        equation_mapping()
    elif tabs.get() == 'Non-Linear Mapping':
        Non_Linear_Mapping()
    elif tabs.get() == 'ACE Filter':
        ace_filter()
    elif tabs.get() == 'Specified Equalization':
        specified_equalization()
    elif tabs.get() == 'Blur Filter':
        threading.Thread(target=Blur_filter()).start()
    elif tabs.get() == 'Denoise Filter':
        threading.Thread(target=Denose_filter()).start()



# ---------------graph functions---------------
def create_graph(root, col, row, img=None, canvas=None, txt='Origional Histogram'):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    fig = Figure(figsize=(5.8, 3.8), dpi=78)
    ax = fig.add_subplot(111)
    ax.plot(hist, color=nor)

    if canvas:
        canvas.get_tk_widget().grid_forget()

    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.set_title(txt)

    for child in root.winfo_children():
        info = child.grid_info()
        if info['row'] == row and info['column'] == col:
            child.destroy()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(column=col, row=row, pady=15, padx=5)

    return canvas


def update_graph(root, r, c, start, start_y, end, end_y):
    if (not start.isdigit() or not start_y.isdigit() or not end.isdigit() or not end.isdigit()) or (start == '' or end == '' or end_y == '' or start_y == '') or (int(start) >= int(end) or int(end_y) < 0 or int(end_y) > 255 or int(start) < 0 or int(end) > 255 or int(start_y) < 0 or int(start_y) > 255):
        messagebox.showerror('Invalid Input', 'Please enter valid input')
        return
    create_mapping_graph(root, r, c, int(start), int(start_y), int(end), int(end_y))


def create_mapping_graph(root, r, c, start=None, start_y=None, end=None, end_y=None):
    global mapping
    if mapping is None:
        mapping = np.arange(255)

    if end is not None or end_y is not None or start is not None or start_y is not None:
        slope = (end_y - start_y) / (end - start)
        mapping[start:end] = start_y + slope * (np.arange(start, end) - start)
    
    fig = Figure(figsize=(5.8, 3.8), dpi=78)
    ax = fig.add_subplot(111)
    ax.plot(mapping, color=nor)
    ax.set_xlabel('Input Pixel Value')
    ax.set_ylabel('Output Pixel Value')
    ax.set_title('Mapping Graph')

    for child in root.winfo_children():
        info = child.grid_info()
        if info['row'] == r and info['column'] == c:
            child.destroy()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(column=c, row=r, pady=5, padx=5)


# ---------------Layout functions---------------
def show_val(value, r, txt):
    global k1_val, k2_val

    if r == 1:
        k1_val =  round(float(value), 2)
        txt.configure(text=f'{k1_val:.2f}')
    else:
        k2_val =  round(float(value), 2)
        txt.configure(text=f'{k2_val:.2f}')

def populize_tab(tab, title):
    global gray_img, arrow

    if title == 'Image Negative':
        global neg_img
        tab.columnconfigure((0,4), weight=1)

        create_graph(tab, 1, 1, img=gray_img, txt='Origional Histogram')
        ctk.CTkLabel(tab, image=arrow ,text='').grid(column=2, row=1, padx=25, pady=10)
        create_graph(tab, 3, 1, txt='Negative Image Histogram')
        
    elif title == 'Shrink & Stretch':
        global Min, Max
        tab.columnconfigure((0,4), weight=1)
        fr = ctk.CTkFrame(tab)
        fr.grid(column=1, row=1, pady=5, padx=5, sticky='news', columnspan=3)
        fr.columnconfigure((0,4), weight=1)
        ctk.CTkLabel(fr, text='New Range: ').grid(column=1, row=0, pady=5, padx=5, sticky='e')
        Min = ctk.CTkEntry(fr, width=105, placeholder_text='Range Min')
        Min.grid(column=2, row=0, padx=5, pady=5)
        Max = ctk.CTkEntry(fr, width=105, placeholder_text='Range Max')
        Max.grid(column=3, row=0, padx=5, pady=5)

        create_graph(tab, col=1, row=0)
        ctk.CTkLabel(tab, image=arrow, text='').grid(column=2, row=0, padx=25, pady=10)
        create_graph(tab, col=3, row=0, txt='Processed Histogram')

    elif title == 'Linear Mapping':
        global fr1
        tab.columnconfigure((0,4), weight=1)

        create_graph(tab, 1, 0)
        ctk.CTkLabel(tab, image=arrow, text='').grid(column=2, row=0, padx=25, pady=10)
        create_graph(tab, 3, 0, txt='Mapped Histogram')

        fr1 = ctk.CTkFrame(tab)
        fr1.grid(column=1, row=1, pady=10, padx=5, sticky='news', columnspan=3)
        fr1.columnconfigure((0,4), weight=1)
        create_mapping_graph(fr1, 0, 2)

        fr = ctk.CTkFrame(tab)
        fr.grid(column=1, row=2, pady=5, padx=5, sticky='news', columnspan=3)
        fr.columnconfigure((0,5), weight=1)
        
        start_entry = ctk.CTkEntry(fr, width=105, placeholder_text='Start Point')
        start_entry.grid(column=1, row=1, padx=5, pady=5)
        start_y_entry = ctk.CTkEntry(fr, width=105, placeholder_text='Start Y Value')
        start_y_entry.grid(column=2, row=1, padx=5, pady=5)

        end_entry = ctk.CTkEntry(fr, width=105, placeholder_text='End Point')
        end_entry.grid(column=3, row=1, padx=5, pady=5)
        end_y_entry = ctk.CTkEntry(fr, width=105, placeholder_text='End Y Value')
        end_y_entry.grid(column=4, row=1, padx=5, pady=5)
        
        but = ctk.CTkButton(fr, text='Create Mapping', command=lambda: update_graph(fr1, 0, 2, start=start_entry.get(), start_y=start_y_entry.get(), end=end_entry.get(), end_y=end_y_entry.get()), fg_color=nor, hover_color=hov)
        but.grid(column=2, row=2, padx=5, pady=5, columnspan=2)

    elif title == 'Non-Linear Mapping':
        global k, r
        tab.columnconfigure((0,4), weight=1)
        create_graph(tab, 1, 0)
        ctk.CTkLabel(tab, image=arrow, text='').grid(column=2, row=0, padx=25, pady=10)
        create_graph(tab, 3, 0, txt='Mapped Histogram')

        fr = ctk.CTkFrame(tab)
        fr.grid(column=1, row=1, pady=5, padx=5, sticky='news', columnspan=3)
        fr.columnconfigure((0,6), weight=1)
        
        eq = ctk.CTkLabel(fr, text='y = K(I(r,c)/K)^r')
        eq.grid(column=1, row=0, padx=5, pady=5, columnspan=4, sticky='n')

        ctk.CTkLabel(fr, text='K Value: ').grid(column=1, row=1, pady=5, padx=5, sticky='e')
        k = ctk.CTkEntry(fr, width=105, placeholder_text='K Value')
        k.grid(column=2, row=1, padx=5, pady=5)

        ctk.CTkLabel(fr, text='r Value (-ve for (1/r)): ').grid(column=3, row=1, pady=5, padx=(10, 5), sticky='e')
        r = ctk.CTkEntry(fr, width=105, placeholder_text='r Value')
        r.grid(column=4, row=1, padx=5, pady=5)

    elif title == 'ACE Filter':
        global l_imgFr, winSize 
        tab.columnconfigure((0,4), weight=1)
        
        create_graph(tab, col=1, row=0,img=gray_img)
        ctk.CTkLabel(tab, image=arrow ,text='').grid(column=2, row=0, padx=25, pady=10)
        create_graph(tab, col=3, row=0, txt='ACE Histogram')

        fr = ctk.CTkFrame(tab)
        fr.grid(column=1, row=2, pady=5, padx=5, sticky='news', columnspan=3)
        fr.columnconfigure((0,4), weight=1)
        
        winSize = ctk.CTkEntry(tab, width=105, placeholder_text='Window Size')
        winSize.grid(column=2, row=1, pady=10, padx=5)

        txt1= ctk.CTkLabel(fr, text='0.50')
        txt1.grid(column=3, row=1, pady=10, padx=5)
        ctk.CTkLabel(fr, text='Gain Factor: ').grid(column=1, row=1, pady=10, padx=5, sticky='e')
        k1 = ctk.CTkSlider(fr, width=300, from_=0, to=1, number_of_steps=100, button_color=nor, progress_color=hov, button_hover_color=hov, command=lambda e: show_val(e,1,txt1))
        k1.grid(column=2, row=1, padx=5, pady=10)
        
        txt2 = ctk.CTkLabel(fr, text='0.50')
        txt2.grid(column=3, row=2, pady=10, padx=5)
        ctk.CTkLabel(fr, text='Retention Factor: ').grid(column=1, row=2, pady=10, padx=5, sticky='e')
        k2 = ctk.CTkSlider(fr, width=300, from_=0, to=1, number_of_steps=100, button_color=nor, progress_color=hov, button_hover_color=hov, command=lambda e: show_val(e,2,txt2))
        k2.grid(column=2, row=2, padx=5, pady=10)
    
    elif title == 'Specified Equalization':
        tab.columnconfigure((0,6), weight=1)

        plus = ctk.CTkImage(Image.open('plus.png'), size=(70, 60))
        noImg = ctk.CTkImage(Image.open('no image.png'), size=image_size)

        f = ctk.CTkFrame(tab)
        f.grid(column=1, row=0, pady=5, padx=5, sticky='news', columnspan=5)
        f.columnconfigure((0,6), weight=1)
        imgFr1 = ctk.CTkFrame(f, corner_radius=20, fg_color="#2B2B2B")
        imgFr1.grid(row=0, column=1, padx=15, pady=15, sticky='snew')
        ctk.CTkLabel(imgFr1, image=noImg, text='').grid(column=0, row=0, padx=10, pady=10)
        ctk.CTkLabel(imgFr1, text='Target Image').grid(column=0, row=1, padx=10, pady=(0,10))

        img1_btn = ctk.CTkButton(f, text='Upload Target Image', command=lambda: select_image(0, 0, imgFr1), fg_color=nor, hover_color=hov)
        img1_btn.grid(column=1, row=2, padx=10, pady=10)

        ctk.CTkLabel(f, image=plus, text='').grid(column=2, row=0, padx=10, pady=10)

        imgFr2 = ctk.CTkFrame(f, corner_radius=20, fg_color="#2B2B2B")
        imgFr2.grid(row=0, column=3, padx=15, pady=15, sticky='snew')
        ctk.CTkLabel(imgFr2, image=noImg, text='').grid(column=1, row=0, padx=10, pady=10)
        ctk.CTkLabel(imgFr2, text='Reference Image').grid(column=1, row=1, padx=10, pady=(0,10))

        img2_btn = ctk.CTkButton(f, text='Upload Reference Image', command=lambda: select_image(1, 0, imgFr2), fg_color=nor, hover_color=hov)
        img2_btn.grid(column=3, row=2, padx=10, pady=10)

        fr = ctk.CTkFrame(tab)
        fr.grid(column=1, row=1, pady=5, padx=5, sticky='news', columnspan=5)
        fr.columnconfigure((0,6), weight=1)
        create_graph(fr, 1, 0, txt='Target Image Histogram')
        ctk.CTkLabel(fr, image=plus, text='').grid(column=2, row=0, padx=20, pady=10)
        create_graph(fr, 3, 0, txt='Reference Image Histogram')
        ctk.CTkLabel(fr, image=arrow, text='').grid(column=4, row=0, pady=25, padx=10)
        create_graph(fr, 5, 0, txt='Processed Histogram')
    
    elif title == 'Blur Filter':
        global winSize1

        tab.columnconfigure((0,6), weight=1)

        create_graph(tab, col=1, row=0, img=gray_img)
        ctk.CTkLabel(tab, image=arrow, text='').grid(column=2, row=0, padx=25, pady=10)
        create_graph(tab, col=3, row=0, txt='Blur Image Histogram')

        fr = ctk.CTkFrame(tab)
        fr.grid(column=1, row=1, pady=5, padx=5, sticky='news', columnspan=3)
        fr.columnconfigure((0,3), weight=1)

        winSize1 = ctk.CTkEntry(fr, width=105, placeholder_text='Window Size')
        winSize1.grid(column=2, row=1, pady=10, padx=5)

    elif title == 'Denoise Filter':
        global winSize2, variance

        tab.columnconfigure((0,6), weight=1)

        create_graph(tab, col=1, row=0, img=gray_img)
        ctk.CTkLabel(tab, image=arrow, text='').grid(column=2, row=0, padx=25, pady=10)
        create_graph(tab, col=3, row=0, txt='Denoised Image Histogram')

        fr = ctk.CTkFrame(tab)
        fr.grid(column=1, row=1, pady=5, padx=5, sticky='news', columnspan=3)
        fr.columnconfigure((0,3), weight=1)

        winSize2 = ctk.CTkEntry(fr, width=105, placeholder_text='Window Size')
        winSize2.grid(column=1, row=1, pady=10, padx=5)

        variance = ctk.CTkEntry(fr, width=105, placeholder_text='Noise Variance')
        variance.grid(column=2, row=1, pady=10, padx=5)



def upper_frame(img=False):
    global app, r_imgFr, l_imgFr, image_size, arrow, upperFr, upload_btn, process_btn

    if app:
        l_imgFr = ctk.CTkFrame(upperFr, corner_radius=20, fg_color="#2B2B2B")
        l_imgFr.grid(row=0, column=1, padx=15, pady=15, sticky='snew')
        
        arrow = ctk.CTkImage(Image.open('right arrow.png'), size=(60, 60))
        ctk.CTkLabel(upperFr, image = arrow, text='').grid(column=3, row=0, padx=10, pady=10)
        r_imgFr = ctk.CTkFrame(upperFr, corner_radius=20, fg_color="#2B2B2B")
        r_imgFr.grid(row=0, column=5, padx=15, pady=15, sticky='snew')
    
        img1 = ctk.CTkImage(Image.open('no image.png'), size=image_size)

        if not img:
            ctk.CTkLabel(l_imgFr, image=img1, text='').grid(column=0, row=0, padx=10,pady=10)
        
        if pro_img is not None:
            ctk.CTkLabel(r_imgFr, image=pro_img, text='').grid(column=0, row=0, padx=10,pady=10)
        else:
            ctk.CTkLabel(r_imgFr, image=img1, text='').grid(column=0, row=0, padx=10,pady=10)
            
        upload_btn = ctk.CTkButton(upperFr, text='Upload Image', command=select_image, fg_color=nor, hover_color=hov)
        upload_btn.grid(column=1, row=1, padx=10, pady=10)

        process_btn = ctk.CTkButton(upperFr, text='Process Image', command=process_image, fg_color=nor, hover_color=hov)
        process_btn.grid(column=5, row=1, padx=10, pady=10)

        ctk.CTkLabel(l_imgFr, text='Original (Gray Scale) Image').grid(column=0, row=1, padx=10, pady=(0,10))
        ctk.CTkLabel(r_imgFr, text='Processed Image').grid(column=0, row=1, padx=10, pady=(0,10))

def basicLayout():
    global app, r_imgFr, l_imgFr, image_size, arrow, tabs, upperFr
    if app:
        app.title('DIP Assignment (21L-1770)')
        width= app.winfo_screenwidth()               
        height= app.winfo_screenheight()
        app.geometry('%dx%d'% (width//1.3, (height//1.3)+40))
        app.columnconfigure(0, weight=1)
        app.rowconfigure(1, weight=1)

        # upper frame for images
        upperFr = ctk.CTkFrame(app, corner_radius=20)
        upperFr.grid(pady=(20, 15), padx=30, row=0, column=0, sticky='new',)
        upperFr.columnconfigure((0,8), weight=1)

        upper_frame()

        # lower frame for controls
        lowerFr = ctk.CTkScrollableFrame(app, corner_radius=20)
        lowerFr.grid(pady=(0, 20), padx=30, row=1, column=0, sticky='snew')
        lowerFr.rowconfigure(0, weight=1)
        lowerFr.columnconfigure(0, weight=1)
        tabs = ctk.CTkTabview(lowerFr, segmented_button_selected_color=nor, segmented_button_unselected_hover_color=hov, segmented_button_selected_hover_color=hov)
        tabs.grid(sticky='news')

        sudomeadian= tabs.add('Blur Filter')
        MMSE       = tabs.add('Denoise Filter')
        ace        = tabs.add('ACE Filter')
        equalize   = tabs.add('Specified Equalization')
        mapping    = tabs.add('Linear Mapping')
        negative   = tabs.add(' Image Negative')
        histogram  = tabs.add('Shrink & Stretch')
        non_linear = tabs.add('Non-Linear Mapping')
        tabs.set('Blur Filter')

        # negative tab content
        populize_tab(negative, 'Image Negative')

        # histogram tab content
        populize_tab(histogram, 'Shrink & Stretch')

        # linear mapping tab content
        populize_tab(mapping, 'Linear Mapping')

        # non linear mapping tab content
        populize_tab(non_linear, 'Non-Linear Mapping')

        # Adaptive Contrast Enhancement tab content
        populize_tab(ace, 'ACE Filter')

        # specified Histogram equalization tab content
        populize_tab(equalize, 'Specified Equalization')

        # Sudomedian Filter tab content
        populize_tab(sudomeadian, 'Blur Filter')

        # MMSE Filter tab content
        populize_tab(MMSE, 'Denoise Filter')


if __name__ == '__main__':
    app = ctk.CTk()
    basicLayout()
    threading.Thread(target=event_handler).start()
    app.mainloop()
    app = None

    if os.path.exists('gray.png'):
        os.remove('gray.png')

    if os.path.exists('ref.png'):
        os.remove('ref.png')

    if os.path.exists('equalized.png'):
        if os.path.exists('specified.png'):
            os.remove('specified.png')
        os.rename('equalized.png','specified.png')
