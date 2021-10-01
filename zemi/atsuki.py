import tkinter
import random
from PIL import Image, ImageTk

window = tkinter.Tk()

# window-size
window.geometry('800x800')

# window-title
window.title('TEST')


# define image
img1 = Image.open('./1.png')
img1 = ImageTk.PhotoImage(img1)

img2 = Image.open('./2.png')
img2 = ImageTk.PhotoImage(img2)

img3 = Image.open('./3.jpg')
img3 = ImageTk.PhotoImage(img3)

img4 = Image.open('./4.jpeg')
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
                            canvas.create_image(400, 50, image=img2, anchor=tkinter.NW)
                              elif img_id == 3:
                                      canvas.create_image(50, 400, image=img3, anchor=tkinter.NW)
                                        elif img_id == 4:
                                                canvas.create_image(400, 400, image=img4, anchor=tkinter.NW)
                                                  else:
                                                          print("error")
                                                            
                                                              # interval
                                                                window.after(4000, draw_image)


                                                                # time to start
                                                                window.after(0, draw_image)
                                                                window.mainloop()
