# importing the tkinter module and PIL
# that is pillow module
from tkinter import *
from PIL import ImageTk, Image
import silhouette as sil

# Calling the Tk (The initial constructor of tkinter)
root = Tk()

# We will make the title of our app as Image Viewer
root.title("Silhouette")

# The geometry of the box which will be displayed
# on the screen
root.geometry("350x520")

image_no_1 = ImageTk.PhotoImage(Image.open("images/portrait.jpg"))
List_images = [image_no_1]

label = Label(image=image_no_1)

# We have to show the box so this below line is needed
label.grid(row=1, column=0, columnspan=3)

def testing():
    sil.test()
    image_no_2 = ImageTk.PhotoImage(Image.open("images/silhouette.jpg"))
    List_images.append(image_no_2)
    label = Label(image=List_images[1])
    label.grid(row=1, column=0, columnspan=3)
    button["state"]=DISABLED

button = Button(root, text="Silhouette", command=testing)

button.grid(row=5, column=0, columnspan=2)

exit = Button(root, text="Exit", command=root.quit)
exit.grid(row=5, column=2)

root.mainloop()