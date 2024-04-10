"""
For fixing the EDF header from NAKO file

the fix_edf() function is from Johannes Zschocke, which replace the Germany 
characters by English characters

the edf_deidentify() function will remove the patient info, recording info and startdate
by X X X X , Startdate X X X X and 01.01.01

author: Yaopeng Ma  sdumyp@126.com or mayaope@biu.ac.il

date: 04/2023
"""

import os
import shutil

def fix_edf(edfpath):
    """
    The function will replace the germany characters by english characters
    NOTE: The modify is on the orignal file !!!
    """
    assert edfpath[-4:] == '.edf', 'Not an EDF file'
    with open(edfpath, 'rb+') as f:
        # read bytes of header
        f.seek(184)
        bytes_header = int(f.read(8))

        f.seek(0)

        for i in range(bytes_header):
            byte = f.read(1)

            # look for ° replace by dot
            if byte == (176).to_bytes(1, byteorder='big'):
                f.seek(i)
                f.write((111).to_bytes(1, byteorder='big'))
                print(f'Position : {i} replaced {byte} with\
                       {str((111).to_bytes(1, byteorder="big"))}')
            # a
            elif byte == (228).to_bytes(1, byteorder='big'):
                f.seek(i)
                f.write((97).to_bytes(1, byteorder='big'))
                print(f'Position : {i} replaced {byte} with\
                       {str((97).to_bytes(1, byteorder="big"))}')

            # o
            elif byte == (246).to_bytes(1, byteorder='big'):
                f.seek(i)
                f.write((111).to_bytes(1, byteorder='big'))
                print(f'Position : {i} replaced {byte} with\
                       {str((111).to_bytes(1, byteorder="big"))}')

            # u
            elif byte == (252).to_bytes(1, byteorder='big'):
                f.seek(i)
                f.write((117).to_bytes(1, byteorder='big'))
                print(f'Position : {i} replaced {byte} with\
                       {str((117).to_bytes(1, byteorder="big"))}')

            # others replace with _
            elif byte > (126).to_bytes(1, byteorder='big'):
                f.seek(i)
                f.write((95).to_bytes(1, byteorder='big'))
                print(f'Position : {i} replaced {byte} with\
                       {str((95).to_bytes(1, byteorder="big"))}')



def edf_deidentify(path, save_dir=None, overwrite=False):
    """

    :param path: path to edf file to be deidentified
    :param save_dir: directory to save deidentified copy of edf (default is directory of edf file)
    :param overwrite: replace the edf file given with the deidentified file (default = False) (Note: if True, ignores
                      save_dir)

    :return: None
    """

    # If no save_dir provided, use dir of edf file
    if save_dir is None:
        save_dir = os.path.dirname(path)
    else:  # check if save_dir is valid
        if not os.path.isdir(save_dir):
            raise Exception("Invalid save_dir path: " + save_dir)

    # Check if invalid file
    if not os.path.isfile(path):
        raise Exception("Invalid file: " + path)

    # Copy file to new name
    if not overwrite:
        path_new = save_dir + '/' + os.path.basename(path)[0:-4] + '_deidentified.edf'
        shutil.copy(path, path_new)
        path = path_new

    # Open file(s) and deidentify
    f = open(path, "r+", encoding="utf-8")  # '' = read
    try:
        f.write('%-8s' % "0")
        f.write('%-80s' % "X X X X")  # Remove patient info
        f.write('%-80s' % "Startdate X X X X")  # Remove recording info
        f.write('01.01.01')  # Set date as 01.01.01
    except UnicodeDecodeError:
        f.close()
        f = open(path, "r+", encoding="iso-8859-2")  # 'r' = read
        try:
            f.write('%-8s' % "0")
            f.write('%-80s' % "X X X X")  # Remove patient info
            f.write('%-80s' % "Startdate X X X X")  # Remove recording info
            f.write('01.01.01')  # Set date as 01.01.01
        except UnicodeDecodeError:
            f.close()
            f = open(path, "r+", encoding="iso-8859-1")  # 'r' = read
            try:
                f.write('%-8s' % "0")
                f.write('%-80s' % "X X X X")  # Remove patient info
                f.write('%-80s' % "Startdate X X X X")  # Remove recording info
                f.write('01.01.01')  # Set date as 01.01.01
                f.close()
            finally:
                raise Exception('No valid encoding format found')

    return


if __name__ == '__main__':
    import os
    folder = r'D:\Aktigraphie\SOMNOwatch plus\EDF'
    filepath = os.path.join(folder, 'SL322_SL322_(1).edf')
    fix_edf(filepath)
