from multiprocessing import Process
import os
from rpeak_detection import save_rpeaks_mad
from ntpath import basename

def process_edfs(edffiles, rri_folder, mad_folder):
    """
    Detect Rpeaks and calculate MAD from EDF file 
    """
    if not os.path.exists(rri_folder):
        os.mkdir(rri_folder)
    if not os.path.exists(mad_folder):
        os.mkdir(mad_folder)
    
    for f in edffiles:
        print("Processing: ", f, end=' ')
        filename = basename(f)
        subjectname = filename.split('.')[0]
        rripath = os.path.join(rri_folder, subjectname + '.rri')
        madpath = os.path.join(mad_folder, subjectname + '_mad.npz')
        if os.path.exists(rripath) and os.path.exists(madpath):
            print(f, "Already did")
            continue

        try:
            save_rpeaks_mad(f, rripath, madpath, True)
            print(f, 'Done')
        except Exception as e:

            with open("./error_message.txt", 'a') as logfile:
                        logfile.write(filename + ';' + str(e) + '\n')
                        print("Process Failed!")


def main():
    num_workers = 30
    folderspath = "/media/yaopeng/data1"
    nakofolders = ["NAKO-33a", "NAKO-33b", 
                   "NAKO-84", "NAKO-84n1", 
                   "NAKO-84n2", "NAKO-419", 
                   "NAKO-419k", "NAKO-609"]
    total = 0
    with open("./error_message.txt", 'w') as f:
        f.write("Error Message\n")

    for folder in nakofolders:
        folder_path = os.path.join(folderspath, folder)
        filenames = os.path.join(folderspath, folder_path)
        filepaths = [os.path.join(folder_path, f) for f in os.listdir(filenames)]
        filepaths = list(filter(lambda x: x[:-4] != '.edf', filepaths))
        process_list = []
        rrifolder = f'RRI/{folder}'
        madfolder = f'MAD/{folder}'
        for i in range(num_workers):
            p = Process(target=process_edfs, 
                        args=(filepaths[i::num_workers], 
                              rrifolder, madfolder)) 
            p.start()
            process_list.append(p)
        
        for p in process_list:
            p.join()
        total += len(filepaths)
    print(total)
if __name__ == "__main__":
    main()
