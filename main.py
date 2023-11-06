from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import string
import numpy as np;
import matplotlib.pyplot as plot;
import matplotlib.pyplot as plt
from math import pi
import os


import fileinput
import heapq
from collections import defaultdict, Counter

#  now we will try to calcualte the frequency of each element.


class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(text):
    # Count the frequency of each character in the text
    freq_count = Counter(text)

    # Create a priority queue (min heap) of Huffman nodes
    heap = [HuffmanNode(char, freq) for char, freq in freq_count.items()]
    heapq.heapify(heap)

    # Build the Huffman tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(None, left.freq + right.freq)
        parent.left = left
        parent.right = right
        heapq.heappush(heap, parent)

    return heap[0]

def build_huffman_codes(node, current_code, codes):
    if node is None:
        return

    if node.char is not None:
        codes[node.char] = current_code
    build_huffman_codes(node.left, current_code + '0', codes)
    build_huffman_codes(node.right, current_code + '1', codes)

def huffman_encode(text):
    if len(text) == 0:
        return None, None

    root = build_huffman_tree(text)
    huffman_codes = {}
    build_huffman_codes(root, '', huffman_codes)

    encoded_text = ''.join(huffman_codes[char] for char in text)
    return encoded_text, huffman_codes

def huffman_decode(encoded_text, huffman_codes):
    if encoded_text is None or huffman_codes is None:
        return None

    reverse_huffman_codes = {code: char for char, code in huffman_codes.items()}
    decoded_text = ''
    current_code = ''
    
    for bit in encoded_text:
        current_code += bit
        if current_code in reverse_huffman_codes:
            decoded_text += reverse_huffman_codes[current_code]
            current_code = ''

    return decoded_text
# !--------------------------------------

def Huffman(s):
    
    if __name__ == '__main__':
        text = s
        encoded_text ,huffman_codes= huffman_encode(text)
        print(f'\nOriginal text: \n{text} \n')
        print(f'\nEncoded text: \n{encoded_text}\n')
        
        decoded_text = huffman_decode(encoded_text, huffman_codes)
        print(f'\nDecoded text: \n{decoded_text}\n')
        return encoded_text,decoded_text
    
#!-----------------------------------------------------------


def ask(s):

    plt.close('all')

    Fs = 1000  # Sampling frequency
    fc = 100   # Carrier frequency


    Td = 0.1              
    samples = int(Td * Fs)  

    binary_data =[]
    for i in s:
        binary_data.append(int(i))

    lengthofdata=len(binary_data)



    t = np.arange(0,lengthofdata/10,1/Fs)

    sig = np.zeros_like(t)

    # making 0 to 99 -->1 , making 100 to 199 -->0 and so on...
    for i, bit in enumerate(binary_data):
        sig[i * samples:(i + 1) * samples] = bit
        
        
        
    x = np.sin(2 * pi * fc * t) # carrier wave





    plt.subplot(2, 1, 1)
    plt.step(t, sig, where='post')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.grid()

    # modulation
    ask = x * sig

    demodulated_data = []
    threshold = 0.5
    print ("Demodulated Data: \n")
    for i in range(0, len(ask), samples):
        phase_mean = np.mean(sig[i:i+samples])
        if phase_mean < threshold:
            demodulated_data.append(0)
        else:
            demodulated_data.append(1)


    plt.subplot(2, 1, 2)
    plt.plot(t, ask)
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.title('ask')
    plt.grid()
    plt.tight_layout()

    demoddata=demodulated_data[0:lengthofdata]

    print(demoddata)
    plt.show() 
    os.system('cls')
# *-------------------------------------------------------------------------------


def fsk(s):
    
    plt.close('all')

    Fs = 1000  # Sampling frequency
    fc = 100   # Carrier frequency


    Td = 0.1              
    samples = int(Td * Fs)  


    binary_data =[]
    for i in s:
        binary_data.append(int(i))

    lengthofdata=len(binary_data)

    t = np.arange(0,lengthofdata/10,1/Fs)

    sig = np.zeros_like(t)

    # making 0 to 99 -->1 , making 100 to 199 -->0 and so on...
    for i, bit in enumerate(binary_data):
        sig[i * samples:(i + 1) * samples] = bit
        
        
        
    x = np.sin(2 * pi * fc * t) # carrier wave




    plt.subplot(2, 1, 1)
    plt.step(t, sig)
    plt.xlabel('time')
    plt.ylabel('amp')
    plt.grid()
    plt.tight_layout()

    #modulation--
    f=fc+fc*(sig/2)
    fsk=np.sin(2*pi*f*t)

    plt.subplot(2, 1, 2)
    plt.plot(t, fsk)
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.title('fsk')
    plt.grid()
    plt.tight_layout()

    #demodulation--
    demodulated_data = []
    print ("Demodulated Data: \n")

    for i in range(0, len(fsk), samples):
        f_mean = np.mean(fsk[i:i+samples] * x[i:i+samples])
        if f_mean < 0:
            demodulated_data.append(1)
        else:
            demodulated_data.append(0)

    demoddata = demodulated_data[0:lengthofdata]
    print(demoddata)

    plt.show()
    os.system('cls')
    

# -------------------------------------------------------------------------------

def psk(s):
    
    plt.close('all')


    Fs = 1000  # Sampling frequency
    fc = 100   # Carrier frequency


    Td = 0.1              
    samples = int(Td * Fs)  

    binary_data =[]
    for i in s:
        binary_data.append(int(i))

    lengthofdata=len(binary_data)



    t = np.arange(0,lengthofdata/10,1/Fs)

    sig = np.zeros_like(t)

    # making 0 to 99 -->1 , making 100 to 199 -->0 and so on...
    for i, bit in enumerate(binary_data):
        sig[i * samples:(i + 1) * samples] = bit
        
        
        
    x = np.sin(2 * pi * fc * t) # carrier wave




    plt.subplot(2, 1, 1)
    plt.step(t, sig)
    plt.xlabel('time')
    plt.ylabel('amp')
    plt.grid()
    plt.tight_layout()

    #modulation
    phase = np.where(sig == 0, np.pi, 0)  # Phase = 0 for '1' and π (180 degrees) for '0'
    Xpsk = np.sin(2 * np.pi * fc * t + phase)

    plt.subplot(2, 1, 2)
    plt.plot(t,Xpsk )
    plt.xlabel('time')
    plt.ylabel('amp')

    plt.grid()

    # Demodulation
    demodulated_data = []

    print ("Demodulated Data: \n")

    threshold = np.pi / 2  

    for i in range(0, len(Xpsk), samples):
        phase_mean = np.mean(phase[i:i+samples])
        if phase_mean < threshold:
            demodulated_data.append(1)
        else:
            demodulated_data.append(0)

    demoddata=demodulated_data[0:lengthofdata]

    print(demoddata)



    plt.show()
    os.system('cls')


# !-----------------------------------------------------------
# !===========================================================

w = Tk()
w.geometry('1080x1920')
w.configure(bg='#141414')


def bttn(x, y, text, bcolor, fcolor, cmd):
    def on_enter(e):
        mybutton['background'] = bcolor
        mybutton['foreground'] = fcolor

    def on_leave(e):
        mybutton['background'] = fcolor
        mybutton['foreground'] = bcolor

    mybutton = Button(w, width=104, height=6, text=text,font=('Helvetica', '15'),
                      fg=fcolor,
                      bg=bcolor,
                      border=2,
                      activeforeground=fcolor,
                      activebackground=bcolor,
                      command=cmd
                      )
    mybutton.bind("<Enter>", on_enter)
    mybutton.bind("<Leave>", on_leave)  # Corrected the event name
    mybutton.place(x=x, y=y)


def cmd1():
    print('You CLicked ASK\n')
    p,q=cmd4()
    ask(q)


def cmd2():
    print('You CLicked FSK\n')
    p,q=cmd4()
    fsk(q)


def cmd3():
    print('You CLicked PSK\n')
    p,q=cmd4()
    psk(q)


def cmd4():
    file=filedialog.askopenfilename(filetypes=[('Text Docs','*.txt')])
    fob=open(file,'r')
    s=fob.read()
    et,dt=Huffman(s)
    st="Encoded Text: \n"+ et + "\n" + "Decoded Text: \n" + dt
    root=Tk()
    root.geometry("600x600")
    label = Label(root, text=st,font="60")
    label.pack()
    binData=et
    return s,et

    
def cmd5():
    p,q=cmd4()
    st="Encoded Text: \n"+ q + "\n" 
    root=Tk()
    root.geometry("600x600")
    label = Label(root, text=st,font="60")
    label.pack()

bttn(155, 0, "ASK", '#82b0fa', "#141414", cmd1)
bttn(155, 200, "FSK", '#7de38c', "#141414", cmd2)

bttn(155, 400, "PSK", '#db5656', "#141414", cmd3)

bttn(155, 600, "HUFFMAN CODE", '#d1c958', "#141414", cmd5)



w.mainloop()

