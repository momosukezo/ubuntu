
import tkinter
import random
from PIL import Image, ImageTk
#import time

window = tkinter.Tk()


#windowsize
window.geometry('800x800')

#windowtitle
window.title('P300Speller')

#define image
#(./1.png')←現在のパスを入力，他のディレクトリも可能？
img1 = Image.open('./zemi/goku.jpg')
img1 = ImageTk.PhotoImage(img1)

img2 = Image.open('./zemi/naruto.jpeg')
img2 = ImageTk.PhotoImage(img2)

img3 = Image.open('./zemi/hanamiti.jpg')
img3 = ImageTk.PhotoImage(img3)

img4 = Image.open('./zemi/rufi.jpg')
img4 = ImageTk.PhotoImage(img4)

# canvas-setting
canvas = tkinter.Canvas(bg = "black", width=1000, height=800)
canvas.place(x=0, y=0)
def draw_image():
	  canvas.delete("all")
	  img_id = random.randrange(1, 5)
	  if img_id == 1:
           canvas.create_image(50, 50, image=img1, anchor=tkinter.NW)
	  elif img_id == 2:
           canvas.create_image(50, 400, image=img2, anchor=tkinter.NW)
	  elif img_id == 3:
	   canvas.create_image(400, 50, image=img3, anchor=tkinter.NW)
	  elif img_id == 4:
           canvas.create_image(400, 400, image=img4, anchor=tkinter.NW)
	  else:
	   print('error')



#interval
#window.after(3000, draw_image)

#time to start
for i in range(5000,0,-500):
  window.after(i, draw_image)

  #windowに表示
window.mainloop()

