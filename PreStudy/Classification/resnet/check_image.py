def test_images(dir):
    import os
    import cv2
    bad_list=[]
    good_list=[]
    good_exts=['jpg', 'png', 'bmp','tiff','jpeg', 'gif'] # make a list of acceptable image file types
    for klass in os.listdir(dir) :  # iterate through the sub directories
        class_path=os.path.join (dir, klass) # create path to sub directory
        if os.path.isdir(class_path):
            for f in os.listdir(class_path): # iterate through image files
                    f_path=os.path.join(class_path, f) # path to image files
                    ext=f[f.rfind('.')+1:] # get the files extension
                    if ext  not in good_exts:
                            print(f'file {f_path}  has an invalid extension {ext}')
                            bad_list.append(f_path)
                    else:
                        try:
                            img=cv2.imread(f_path)
                            size=img.shape
                            good_list.append(f_path)
                        except:
                            print(f'file {f_path} is not a valid image file ')
                            bad_list.append(f_path)
        else:
            print(f'** WARNING ** directory {dir} has files in it, it should only contain sub directories')

    return good_list, bad_list