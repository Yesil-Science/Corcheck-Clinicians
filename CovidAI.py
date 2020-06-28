#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pydicom
import math
from tkinter import *
from tkinter import filedialog
from PIL import Image,ImageTk
import os
from google_drive_downloader import GoogleDriveDownloader as gdd
from keras.models import load_model
import efficientnet.keras as efn
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as k
import logging
logging.getLogger('tensorflow').disabled = True
cwd = os.getcwd()


# In[2]:


print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")
print("YAPAY ZEKA MODELİ YÜKLENİYOR LÜTFEN BEKLEYİNİZ....")


if not(os.path.exists("Covid.h5")):
    gdd.download_file_from_google_drive(file_id='1MKL7V-AXVHKWXFruGN1xI2eZmkF6roAn',dest_path= cwd + "\Covid.h5",unzip=False)
    gdd.download_file_from_google_drive(file_id='1H0YFK6Gg7SnAaemGmmTD5ww6wo6oCZoP',dest_path= cwd + "\covidct3d_model.h5",unzip=False)                                    
                                    
model=load_model(r"Covid.h5")
modelct=load_model(r"covidct3d_model.h5")


# In[3]:


#Funcs for CT preprocessing

def translate(pred):
        if pred[0][0][0]>=0.5:
            return "Healthy"
            #return (" Sağlıklı({})".format(round(pred[0][0][0])))
        else:
            return "Covid"
            #return("Hasta({})".format(1-round(pred[0][0][0])))
        


def normalize2(image):
    mean = image.mean()
    if mean >0:
        MIN_BOUND = 0
        MAX_BOUND = 1024
    else:
        MIN_BOUND = -1024
        MAX_BOUND = 0
        
    image[image == -2000] = 0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    
    image[image>1] = 1.
    image[image<0] = 0.
    return image




def chunks(l, n):
    count = 0
    for i in range(0, len(l), n):
        if (count < NoSlices):
            yield l[i:i + n]
            count = count + 1


def mean(l):
    return sum(l) / len(l)


def unlabeledProcessing(path,size=144, noslices=30):
    path = path
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    new_slices = []
    slices = [normalize2(cv2.resize(np.array(each_slice.pixel_array), (size, size))) for each_slice in slices]
    
    chunk_sizes = math.floor(len(slices) / noslices)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)
    if not new_slices:
        print("Could not create test set")
    else:
        
        return np.expand_dims(np.resize(np.array(new_slices),(30,144,144,1)),axis=0)




# In[4]:



def resol(x):
    a =cv2.imread(x)
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    image=cv2.resize(a,(300,300))
    c=np.expand_dims(image,0)
    return(efn.preprocess_input(c))




def analiz(x): #buradaki fonksiyon resim directory alıyor ve yz ile analiz edip ısı haritalı resmi bulunduğu dosyaya kaydediyor
    global resultimg,panel
    sonuc=""
    pred2=model.predict(resol(x))
    ind=np.argmax(pred2[0])
    if ind==0:
        sonuc="Covid"
    if ind==1:
        sonuc="Normal"
    if ind==2:
        sonuc="Pnemio"
    
    vector=model.output[:,ind]
    last_conv=model.get_layer("top_conv")
    grads=k.gradients(vector,last_conv.output)[0]
    pooled_grad=k.mean(grads,axis=(0,1,2))
    iterate=k.function([model.input],[pooled_grad,last_conv.output[0]])
    pooled_grad_value,conv_layer_value=iterate([resol(x)])
    for i in range(1536):
        conv_layer_value[:,:,i] *= pooled_grad_value[i]
    heatmap=np.mean(conv_layer_value,axis=-1)

    plt.rcParams["figure.figsize"]=(16,8)
    img=cv2.imread(x)
    img=cv2.resize(img,(300,300))
    heatmap=np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)
    heatmap=cv2.resize(heatmap,(img.shape[1],img.shape[0]))
    heatmap=np.uint8(255*heatmap)
    i=7
    heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    z=heatmap*0.4+img
    cv2.imwrite('xrayresult.jpg',z)
    resultimg = ImageTk.PhotoImage(Image.open("xrayresult.jpg"))
    panel = Label(root, image=resultimg)
    return(sonuc)

sonuclarct=[]


# In[5]:


NoSlices = 30
def diagnose():                      #Creates diagnose page 
    widget_list = all_children(root)
    for item in widget_list:
        item.place_forget()
    corz.place(relx=.0,rely=.0)
    ctz.place(relx=.34,rely=.65)
    homez.place(relx=.0,rely=.2)
    progz.place(relx=.0,rely=.3)
    diagz_clicked.place(relx=.0,rely=.4)
    diagfirstLabel.place(relx=.4,rely=.26)
    diagsecondLabel.place(relx=.4,rely=.32)
    xrayz.place(relx=.23,rely=.65)
    ctz.place(relx=.6,rely=.65)
    skipz.place(relx=.5,rely=.8)
    
    
def all_children (window) :               #Function for deleting all widgets
    _list = window.winfo_children()

    for item in _list :
        if item.winfo_children() :
            _list.extend(item.winfo_children())

    return _list


def progpage():                            #Creates Prognosis page
    widget_list = all_children(root)
    for item in widget_list:
        item.place_forget()
    corz.place(relx=.0,rely=.0)
    progfirstLabel.place(relx=.35,rely=.26)
    progsecondLabel.place(relx=.35,rely=.32)
    ctzp.place(relx=.4,rely=.65)
    homez.place(relx=.0,rely=.2)
    progz_clicked.place(relx=.0,rely=.3)
    diagz.place(relx=.0,rely=.4)
    skipz3.place(relx=.5,rely=.8)
    
    


def homef():                            #Creates Home page 
    widget_list = all_children(root)
    for item in widget_list:
        item.place_forget()
    corz.place(relx=.0,rely=.0)
    corz2.place(relx=.35,rely=.2)
    progbz.place(relx=.6,rely=.65)
    diagbz.place(relx=.23,rely=.65)
    homez_clicked.place(relx=.0,rely=.2)
    progz.place(relx=.0,rely=.3)
    diagz.place(relx=.0,rely=.4)


sonuclarxray=[]     
img_names=[]  


def addimage(x): 
    #Opening Image Files
    
    if len(img_names)==1:
        img_names.clear()
    
    widget_list = all_children(root)
    for item in widget_list:
        item.place_forget()
        
    corz.place(relx=.0,rely=.0)
    homez.place(relx=.0,rely=.2)
    progz.place(relx=.0,rely=.3)
    diagz.place(relx=.0,rely=.4)
    uploadingLabel.place(relx=.35,rely=.4)
    
    if x == "xray":
        filename=filedialog.askopenfilename(initialdir="/", title="Select Xray Image",filetypes=(("jpeg files","*.jpg"),("png files","*.png")))
        img_names.append(filename)
        xrayprediction = None
        if len(filename):
            xrayprediction = analiz(img_names[0])
            sonuclarxray.append(xrayprediction)
            nextxray.place(relx=.4,rely=.77)
            
    elif x == "ctdiag":
        filename=filedialog.askdirectory()
        img_names.append(filename)
        if len(filename):
            ctprediction =modelct.predict(unlabeledProcessing(img_names[0]))
            sonuclarct.append(ctprediction)
            nextctnormal.place(relx=.4,rely=.77)
    elif x == "ctprog":
        filename=filedialog.askdirectory()
        img_names.append(filename)
        if len(filename):
            ctprediction =modelct.predict(unlabeledProcessing(img_names[0]))
            sonuclarct.append(ctprediction)
            nextctbscore.place(relx=.4,rely=.77)



def result(page):                       #Creates Results Page 
    puan=0   
    widget_list = all_children(root)
    for item in widget_list:
        item.place_forget()
        
    corz.place(relx=.0,rely=.0)
    homez.place(relx=.0,rely=.2)
    progz.place(relx=.0,rely=.3)
    diagz.place(relx=.0,rely=.4)
    resultcanvas.place(relx=.33,rely=.15)
    takez.place(relx=.5,rely=.77)
    resLabel.place(relx=.4,rely=.1)
    
    

    #
    if sonuclarxray:
        h = Label(resultcanvas,bg="#909090", fg="white",font =("Montserrat", 14), text=str(sonuclarxray[0]))
        h.place(relx=.24,rely=.5)
        panel.place(relx=.60,rely=.25)
       
    if sonuclarct:
        c = Label(resultcanvas,bg="#909090", fg="white",font =("Montserrat", 14), text=str(translate(sonuclarct)))
        c.place(relx=.24,rely=.5)
    sonuclarct.clear()
    sonuclarxray.clear()
        
        
    #Normal semptom part 
    if page == 1:
        radioLabel.place(relx=.05,rely=.5)
        sympLabel.place(relx=.05,rely=.57)
        riskLabel.place(relx=.05,rely=.3)
        if temas.get()==1:
            puan=puan+3
     
        if ates.get()==1:
            puan=puan+1
    
        if oksuruk.get()==1:
            puan=puan+1
    
        if nefes.get()==1:
            puan=puan+2
    
        if hastalık.get()==1:
            puan=puan+1
    
        if ates.get()==1 and hastalık.get()==1:
            puan=puan+3
    
        if hastalık.get()==1 and oksuruk.get()==1:
            puan=puan+3
        if puan>=3:
            risklabel1.place(relx=.2,rely=.57)
        if puan==2:
            risklabel2.place(relx=.2,rely=.57)
        if puan==1:
            risklabel3.place(relx=.2,rely=.57)
            


        #resultcanvas.create_image(50, 10, image=gif1, anchor=NW)
            
    
    
    # Brescia check part
    elif page ==2:
        tedaviz="Remdesivir (Eğer mevcut değilse:  Lopinavir/Ritonavir veya Boosted/Darunavir) +  Klorokin veya Hidroksiklorin + Deksametazonu Düşün  (Yaş, komorbidite ve bilişsel bozulma göz önüne alınarak) + Tocilizumab'ı Düşün. (Dahil etme kriterleri: 1. Yüksek SARS-CoV2 viral yükü riskinin düşük olması: >72 saat ateş olmaması, semptomların >7 gündür olması 2.Interlökin-6>40pg/ml veya yüksek plazmatik D-Dimer ve/veya yüksek CRP ve/veya  yüksek ferritin ve/veya yüksek ﬁbrinojen  seviyesi)"
        cevab = ""
        tedavi=""
        if (solunumd.get() + solunumsayid.get() + grafid.get() + oksijend.get())==0:
            cevab ="Hastayı SpO2 ile monitorize etmeye devam et ve klinik olarak değerlendir"
            tedavi = "Standart Tedavi Lopinavir / Ritonavir (veya Boosted/Darunavir) + Klorokin veya Hidroksiklorokin"

        if (solunumd.get() + solunumsayid.get() + grafid.get() + oksijend.get())==1:
            cevab ="Oksijen desteği ekle. Hastayı SpO2 ile monitorize etmeye devam et ve klinik olarak değerlendir"
            tedavi = "Standart Tedavi Lopinavir / Ritonavir (veya Boosted/Darunavir)  Klorokin veya Hidroksiklorokin"
        if (solunumd.get() + solunumsayid.get() + grafid.get() + oksijend.get())==2:
            cevab ="Göğüs grafisi çek, gaz analizi yap Oksijen desteği ekle Hastayı SpO2 ile monitorize etmeye devam et ve klinik olarak değerlendir"
            tedavi = "Standart Tedavi + Deksametazonu düşün (Yaş, komorbidite ve bilişsel bozulma göz önüne alınarak)"

        if (solunumd.get() + solunumsayid.get() + grafid.get() + oksijend.get())>2:
            cevab ="Her 2 günde bir göğüs grafisi çek ve günde 2 kere gaz analizi yap. Hastayı SpO2 ile monitorize etmeye devam et ve klinik olarak değerlendir"
            tedavi = "Standart Tedavi Deksametazonu düşün (Yaş, komorbidite ve bilişsel bozulma göz önüne alınarak) Tocilizumab'ı düşün (Dahil etme kriterleri: 1. Yüksek SARS-CoV2 viral yükü riskinin düşük olması: >72 saat ateş olmaması semptomların >7 gündür olması 2.Interlökin-6>40pg/ml veya yüksek plazmatik DDimer ve/veya yüksek CRP ve/veya yüksek ferritin ve/veya yüksek ﬁbrinojen seviyesi)"

        
            if prone.get()==1:
                cevab = "Yüksek derecede karmaşıklık"
                tedavi = "Remdesivir (Eğer mevcut değilse:  Lopinavir/Ritonavir veya Boosted/Darunavir) +  Klorokin veya Hidroksiklorin + Deksametazonu Düşün  (Yaş, komorbidite ve bilişsel bozulma göz önüne alınarak) + Tocilizumab'ı Düşün.  (Dahil etme kriterleri: 1. Yüksek SARS-CoV2 viral yükü  riskinin düşük olması: >72 saat ateş olmaması, semptomların >7 gündür olması 2.Interlökin-6>40pg/ml veya yüksek plazmatik D-Dimer  ve/veya yüksek CRP ve/veya  yüksek ferritin ve/veya yüksek ﬁbrinojen  seviyesi)"
            
            elif cpap.get()==0:
                tedavi = "Cpap devam ve Standart Tedavi Deksametazonu düşün (Yaş, komorbidite ve bilişsel bozulma göz önüne alınarak) Tocilizumab'ı düşün (Dahil etme kriterleri: 1. Yüksek SARS-CoV2 viral yükü riskinin düşük olması: >72 saat ateş olmaması semptomların >7 gündür olması 2.Interlökin-6>40pg/ml veya yüksek plazmatik DDimer ve/veya yüksek CRP ve/veya yüksek ferritin ve/veya yüksek ﬁbrinojen seviyesi)" 
                
            elif cmv.get()==0:
                cevab = "Dahili Bir Weaning (Mekanik Ventilasyonun Sonlandırılması) Protokolü Kullanarak Hastayı Yoğun Bakımda Tut"
                tedavi = tedaviz

            elif pao.get()==0:
                cevab="Sedasyonu minimize etmeyi dene(RASS- RichmondAjitasyon Sedasyon Skalası-1 ile 0 ) Günlük SBT(SpontanBreathing Trial) uygula."
                tedavi = tedaviz
            elif nmba.get()==0:
                cevab = "Sedasyonu minimize etmeyi dene(RASS- RichmondAjitasyon Sedasyon Skalası-1 ile 0 )"
                tedavi = tedaviz
            elif prone.get()==0:
                cevab = "En iyi PEEP(Positive End Expratory Pressure) ve Kompliyans hesaplamasını yap. NMBA'yi askıya almayı dene Volum seviyesini optimize et."
                tedavi = tedaviz
            
        bLabel.place(relx=.05,rely=.3)
        sugLabel.place(relx=.05,rely=.5)
        progzlabel = Label(resultcanvas,bg="#909090", fg="#5A2892",font=("Montserrat", 13), text=cevab, wraplength = 600, anchor=W, justify=LEFT)
        progzlabel.place(relx=.07,rely=.38)
        progcevab = Label(resultcanvas,bg="#909090", fg="white",font=("Montserrat", 13), text=tedavi, wraplength = 600, anchor=W, justify=LEFT)
        progcevab.place(relx=.07,rely=.6)
        radioLabel.place(relx=.05,rely=.2)
        try: 
            c.place(relx=.24,rely=.2)
        except:
            pass
        
    
    ates.set(0)
    oksuruk.set(0)
    temas.set(0)
    nefes.set(0)
    hastalık.set(0)
    solunumd.set(0)
    solunumsayid.set(0)
    grafid.set(0)
    oksijend.set(0)
    cpap.set(0)
    cmv.set(0)
    pao.set(0)
    nmba.set(0)
    prone.set(0)
    

def sempf():                                #Creates Symptoms Page
    widget_list = all_children(root)
    for item in widget_list:
        item.place_forget()
    corz.place(relx=.0,rely=.0)
    homez.place(relx=.0,rely=.2)
    progz.place(relx=.0,rely=.3)
    diagz.place(relx=.0,rely=.4)
    resultcanvas.place(relx=.33,rely=.15)
    sympLabel.place(relx=.05,rely=.1)
    nextiz.place(relx=.35,rely=.77)
    skipz2.place(relx=.65, rely=.79)
    atesb.place(relx=.1, rely=.17)
    oksurukb.place(relx=.1, rely=.27)
    nefesb.place(relx=.1, rely=.37)
    hastalıkb.place(relx=.1, rely=.47)
    temasb.place(relx=.1, rely=.87)
    
    
    
def Bscore():                              #Creates Brescia form page
    widget_list = all_children(root)
    for item in widget_list:
        item.place_forget()
    corz.place(relx=.0,rely=.0)
    homez.place(relx=.0,rely=.2)
    progz.place(relx=.0,rely=.3)
    diagz.place(relx=.0,rely=.4)
    resultcanvas.place(relx=.33,rely=.15)
    skipz4.place(relx=.55, rely=.77)
    solunum.place(relx=.1, rely=.17)
    solunum_sayi.place(relx=.1, rely=.27)
    oksijen.place(relx=.1, rely=.37)
    grafi.place(relx=.1, rely=.47)
    bresczb.place(relx=.1, rely=.77) #button for semp
    
    
counts=1
def control():

    global counts
    
    if counts ==1:
        if (solunumd.get() + solunumsayid.get() + grafid.get() + oksijend.get())>2:
            oksijen.place_forget()
            grafi.place_forget()
            solunum.place_forget()
            solunum_sayi.place_forget()
            secondp.place(relx=.1, rely=.3)
            counts=counts+1
        else: res()
    
    elif counts==2:
        if cpap.get()==1:
            secondp.place_forget()
            entübe.place(relx=.07, rely=.3)
            counts=counts+1
        else: res()
        
      
    elif counts ==3:
        if cpap.get()==1:
            entübe.place_forget()
            cmvb.place(relx=.1, rely=.3)
            counts=counts+1
        else: res()
            
    elif counts ==4:
        if cmv.get()==1:
            cmvb.place_forget()
            paob.place(relx=.1, rely=.3)
            counts=counts+1
        else: res()
                
    elif counts ==5:
        if pao.get()==1:
            paob.place_forget()
            nmbab.place(relx=.1, rely=.3)
            counts=counts+1
        else: res()
            
    elif counts ==6:
        if nmba.get()==1:
            nmbab.place_forget()
            proneb.place(relx=.1, rely=.3)
            counts=counts+1
        else: res()
            
    elif counts ==7:
        if prone.get()==1:
            proneb.place_forget()
            bresczb.place_forget()
            answerlabel.place(relx=.35, rely=.3)
            nextiz2.place(relx=.35,rely=.75)
            counts = 1
        else: res()
        
def res():
    global counts 
    secondp.place_forget()
    oksijen.place_forget()
    grafi.place_forget()
    solunum.place_forget()
    solunum_sayi.place_forget()
    cmvb.place_forget()
    paob.place_forget()
    nmbab.place_forget()
    proneb.place_forget()
    answerlabel.place(relx=.2, rely=.3)
    bresczb.place_forget()
    nextiz2.place(relx=.35,rely=.75)
    counts = 1


# In[6]:


root = Tk()
root.geometry("1920x1080")
root.configure(background="#4B4D60")
root.title("Covid AI")
root.iconbitmap('ico1.ico')


#Symptoms
ates = IntVar()
oksuruk=IntVar()
temas=IntVar()
nefes=IntVar()
hastalık=IntVar()

#Brescia semp
solunumd = IntVar()
solunumsayid = IntVar()
grafid = IntVar()
oksijend = IntVar()
cpap = IntVar()
cmv = IntVar()
pao = IntVar()
nmba = IntVar()
prone= IntVar()

     
#Photos

diagb = PhotoImage(file="assets\diagb.png") # Big diagnose button img
progb = PhotoImage(file="assets\progb.png")

xrayimg = PhotoImage(file="xrayimg.png")
ctimg = PhotoImage(file="assets\ctimg.png")

corimg = PhotoImage(file="assets\cclogo.png")
corimg2 = PhotoImage(file="assets\CB2.png")

home = PhotoImage(file="assets\home.png")
prog = PhotoImage(file="assets\prog.png")
diag = PhotoImage(file="assets\diag.png")

home_clicked = PhotoImage(file="assets\home_clicked.png")
prog_clicked = PhotoImage(file="assets\prog_clicked.png")
diag_clicked = PhotoImage(file="assets\diag_clicked.png")

skip = PhotoImage(file="assets\skip.png")
nexti = PhotoImage(file="assets\next.png")
take = PhotoImage(file="assets\takeagain.png")
nextQ = PhotoImage(file="assets\nextQ.png")

#canvas
resultcanvas = Canvas(root,height=420 ,width=750 ,bg="#909090",borderwidth=0,highlightthickness=0)


#Bresica questions
solunum = Checkbutton(resultcanvas,variable=solunumd, fg="#2D3243",bg="#909090",font=("Montserrat", 14),text="Hastanın solunum sıkıntı veya kesik kesik konuşması var mı?",)
solunum_sayi = Checkbutton(resultcanvas,variable=solunumsayid,fg="#2D3243", bg="#909090",font=("Montserrat", 14), text="Solunum sayısı >22 mi?")
oksijen = Checkbutton(resultcanvas, variable=oksijend,fg="#2D3243", bg="#909090",font=("Montserrat", 14),text="PaO2 <65mmHg yada SpO2 <%90 mı?")
grafi = Checkbutton(resultcanvas,variable=grafid,fg="#2D3243", bg="#909090",font=("Montserrat", 14), text="Göğüs grafisinde belirgin kötüleşme var mı?")
secondp = Checkbutton(resultcanvas,variable=cpap,fg="#2D3243", bg="#909090",font=("Montserrat", 14),text="CPAP kullanılmasına rağmen hala 2'den fazla kriter pozitif mi?",wraplength = 600, anchor=W, justify=LEFT)
cmvb = Checkbutton(resultcanvas,variable=cmv, fg="#2D3243", bg="#909090",font=("Montserrat", 14),text="CMV (Continuous Mandatory Ventilation)?")
paob = Checkbutton(resultcanvas,variable=pao,  fg="#2D3243",bg="#909090",font=("Montserrat", 14),text="PaO2/FiO2<150mmHg? (Parsiyel O2 basıncı/ Fraction of Inspired Oxygen)")
nmbab = Checkbutton(resultcanvas,variable=nmba, fg="#2D3243", bg="#909090",font=("Montserrat", 14), text="NMBA (Neuromuscular blockade)?")
proneb = Checkbutton(resultcanvas,variable=prone, fg="#2D3243", bg="#909090",font=("Montserrat", 14), text="Yüzüstü pozisyon? iNO(inhaled Nitrik Oksit)? ECLS(Extracorporeal Life Support)?",wraplength = 600, anchor=W, justify=LEFT)


#Semp questions
atesb= Checkbutton(resultcanvas, bg="#909090", fg="#2D3243",font=("Montserrat", 14),text="Ateşim 38C üstünde",variable=ates)
oksurukb= Checkbutton(resultcanvas, bg="#909090", fg="#2D3243",font=("Montserrat", 14),text="Öksürüğüm var",variable=oksuruk)
nefesb= Checkbutton(resultcanvas, bg="#909090", fg="#2D3243",font=("Montserrat", 14),text="Nefes darlığım var ",variable=nefes)
hastalıkb= Checkbutton(resultcanvas, bg="#909090", fg="#2D3243",font=("Montserrat", 14),text="Aşağıdaki hastalık veya durumlardan herhangi birine sahibim \n-İmmun yetmezlik \n-Hipertansiyon \n-Kemoterapi \n-Diyabet\n-Astım  ",variable=hastalık)
temasb= Checkbutton(resultcanvas, bg="#909090", fg="#2D3243",font=("Montserrat", 14),text="COVID-19 için test yapılmış ve sonucu pozitif gelmiş kişilerle temasım oldu",variable=temas)


#Labels
entübe = Label(resultcanvas, text="Lütfen Hastayı entübe ediniz", fg="white", bg="#909090", font=("Montserrat", 23))
answerlabel=Label(resultcanvas, bg="#909090",font=("Montserrat", 16), text="Cevaplarınız Kaydedilmiştir Lütfen \n Sonraki Sayfaya Geçiniz",fg="white")
resLabel = Label(resultcanvas, text="RESULT", fg="#2D3243", bg="#909090", font=("Montserrat", 16))
riskLabel = Label(resultcanvas, text="Risk of COVID-19:", fg="#2D3243", bg="#909090", font=("Montserrat", 16))
radioLabel = Label(resultcanvas, text="Radiological:", fg="#2D3243", bg="#909090", font=("Montserrat", 16))
sympLabel = Label(resultcanvas, text="Symptoms:", fg="#2D3243", bg="#909090", font=("Montserrat", 16))
diagfirstLabel = Label(root, text="First Step: Upload an X-Ray or CT image",fg="white", bg="#4B4D60",font=("Montserrat", 16))
diagsecondLabel = Label(root, text="Second Step: Answer symptom questions",fg="white", bg="#4B4D60",font=("Montserrat", 16))
progfirstLabel = Label(root, text="First Step: Upload CT image",fg="white", bg="#4B4D60",font=("Montserrat", 16))
progsecondLabel = Label(root, text="Second Step: Answer questions for clinical findings \n(BRESCIA score)",fg="white", bg="#4B4D60",font=("Montserrat", 16))
uploadedLabel = Label(root, text="Files uploaded, press next for second step",fg="white", bg="#4B4D60",font=("Montserrat", 16))
uploadfail = Label(root, text="File prediction failed, please try again",fg="white", bg="#4B4D60",font=("Montserrat", 16))
uploadingLabel = Label(root, text="Files are being uploaded and processed. \n This might take a few minutes. \n \n Please Wait",fg="white", bg="#4B4D60",font=("Montserrat", 16))

bLabel = Label(resultcanvas,bg="#909090", fg="#2D3243",font=("Montserrat", 14), text="Berscia Score Result:")
sugLabel = Label(resultcanvas,bg="#909090", fg="#2D3243",font=("Montserrat", 14), text="Suggestions:")
    
#risk labels for semptoms
risklabel1 = Label(resultcanvas,bg="#909090",font=("Montserrat", 15), text="High risk",fg="white")
risklabel2 = Label(resultcanvas,bg="#909090",font=("Montserrat", 15), text="Normal risk",fg="white")
risklabel3 = Label(resultcanvas,bg="#909090",font=("Montserrat", 15), text="Low risk",fg="white")

#Buttons
bresczb = Button(resultcanvas, image = nextQ, command = control,borderwidth=0,highlightthickness=0)

nextiz = Button(root, image = nexti, command = lambda:result(1),borderwidth=0,highlightthickness=0)

nextiz2 = Button(root, image = nexti, command = lambda:result(2),borderwidth=0,highlightthickness=0)
nextxray = Button(root, image = nexti, command = sempf,borderwidth=0,highlightthickness=0)
nextctnormal = Button(root, image = nexti, command = sempf,borderwidth=0,highlightthickness=0)
nextctbscore = Button(root, image = nexti, command = Bscore,borderwidth=0,highlightthickness=0)

takez = Button(root, image = take, command = homef,borderwidth=0,highlightthickness=0)

homez = Button(root, image = home, command = homef,borderwidth=0,highlightthickness=0)
homez.place(relx=.0,rely=.2)

progz = Button(root, image = prog,command=progpage,borderwidth=0,highlightthickness=0)
progz.place(relx=.0,rely=.3)

diagz = Button(root, image = diag , command=diagnose,borderwidth=0,highlightthickness=0)
diagz.place(relx=.0,rely=.4)

homez_clicked = Button(root, image = home_clicked, command = homef,borderwidth=0,highlightthickness=0)
progz_clicked = Button(root, image = prog_clicked, command=progpage,borderwidth=0,highlightthickness=0)
diagz_clicked = Button(root, image = diag_clicked, command=diagnose,borderwidth=0,highlightthickness=0)

skipz = Button(root, image = skip, command = sempf,borderwidth=0,highlightthickness=0)
skipz2 = Button(root, image = skip, command = lambda:result(1),borderwidth=0,highlightthickness=0)
skipz3 = Button(root, image = skip, command = Bscore,borderwidth=0,highlightthickness=0)
skipz4 = Button(root, image = skip, command = lambda:result(2),borderwidth=0,highlightthickness=0)

corz = Label(root, image = corimg,borderwidth=0,highlightthickness=0)
corz.place(relx=.0,rely=.0)

corz2 = Label(root, image = corimg2,borderwidth=0,highlightthickness=0)
corz2.place(relx=.35,rely=.2)

progbz = Button(root,image=progb,command = progpage,borderwidth=0,highlightthickness=0)
progbz.place(relx=.6,rely=.65)

diagbz = Button(root, image = diagb,command=diagnose,borderwidth=0,highlightthickness=0)
diagbz.place(relx=.23,rely=.65)

xrayz = Button(root,image= xrayimg, command= lambda:addimage("xray"),borderwidth=0,highlightthickness=0)
#xrayz.place(relx=.6,rely=.65)

ctz = Button(root, image = ctimg, command= lambda:addimage("ctdiag"),borderwidth=0,highlightthickness=0)
ctzp = Button(root, image = ctimg, command= lambda:addimage("ctprog"),borderwidth=0,highlightthickness=0)

 
root.mainloop()


# In[ ]:




