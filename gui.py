import os
from skimage.color.colorconv import gray2rgb, rgb2gray
import skimage.io as io1
import io
import PySimpleGUI as sg
from matplotlib import pyplot as plt
from PIL import Image
from daugman import process_iris
from hamming_distance import HammingDist
file_types = [("BMP (*.bmp)", "*.bmp"), ("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]


def main():
    layout = [
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Prepare Database"),
            sg.Button("Add a new image to database"),
            sg.Button("Load Image to check"),
        ],
        [sg.Image(key="-ORG_IMAGE-"), sg.Image(key="-IRIS_IMAGE-")],
        [sg.Image(key="-NORMALIZED_IMAGE-"), sg.Image(key="-TEMPLATE_IMAGE-")],
        [sg.Text(key="-Result-")],
        [sg.Text(key="-Error-")],
    ]
    window = sg.Window("Iris detection System", layout)
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Prepare Database":
            # read imgs from folder images and apply procees_iris(filename)
            window["-Result-"].update("Database preparing..")
            # get all files in folder
            files = os.listdir("database")
            # get all files with .bmp extension
            files = [f for f in files if f.endswith(".bmp")]
            # read all files
            for file in files:
                # read image
                img = Image.open("database/" + file)
                # process image
                code1, mask1, polar_arr, imgwithnoise = process_iris(
                    'database/' + file)
                plt.imsave('templates/' + file, code1, cmap='gray')
                plt.imsave('masks/' + file, mask1, cmap='gray')
            # show message
            window["-Result-"].update("Database prepared...")
        elif event == "Add a new image to database":
            filename = values["-FILE-"]
            # if file is none ignore event

            if filename is None:
                continue
            if filename == "":
                continue
            # check if filename exists in database directory
            if os.path.exists("database/" + os.path.basename(filename)):
                window["-Error-"].update("File already exists in database")
                window["-FILE-"].update('')
                continue
            image = Image.open(filename)
            image = image.resize((256, 256))
            image.save("database/" + os.path.basename(filename))
            window["-FILE-"].update('')
        elif event == "Load Image to check":
            filename = values["-FILE-"]
            if os.path.exists(filename):
                image = Image.open(filename)
                image.thumbnail((400, 400))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-ORG_IMAGE-"].update(data=bio.getvalue())
                code1, mask1, polar_arr, imgwithnoise = process_iris(filename)
                code1 = gray2rgb(code1)
                mask1 = gray2rgb(mask1)
                window["-FILE-"].update('')
                # save numpy.ndarray to image file
                plt.imsave('iris_output.bmp', imgwithnoise)
                plt.imsave('template.bmp', code1)
                plt.imsave('normalized.bmp', polar_arr)
                image = Image.open('images/ss.bmp')
                image.thumbnail((400, 400))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IRIS_IMAGE-"].update(data=bio.getvalue())

                image = Image.open('normalized.bmp')
                image.thumbnail((400, 400))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-NORMALIZED_IMAGE-"].update(data=bio.getvalue())

                image = Image.open('template.bmp')
                image.thumbnail((400, 400))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-TEMPLATE_IMAGE-"].update(data=bio.getvalue())

                templates = os.listdir("templates")
                # get all files with .bmp extension
                templates = [f for f in templates if f.endswith(".bmp")]

                masks = os.listdir("masks")
                # get all files with .bmp extension
                masks = [f for f in masks if f.endswith(".bmp")]
                hd=10000
                for template in templates:
                    for mask in masks:
                        print(template, mask)
                        if template == mask:
                            print(template, mask)
                            code2 = io1.imread('templates/' + template)
                            mask2 = io1.imread('masks/' + mask)
                            mask2 = mask2.astype(bool)
                            hd = HammingDist(code1, mask1, code2, mask2)
                            print(hd)
                            if(hd <= .4):
                                break

                if hd <= .4:
                    window["-Result-"].update("Search ends with iris found ")
                else:
                    window["-Result-"].update("Search ends with no iris found")

    window.close()


if __name__ == "__main__":
    main()